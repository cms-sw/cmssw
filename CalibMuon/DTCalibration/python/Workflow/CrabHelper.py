from __future__ import print_function
from __future__ import absolute_import
import logging
import sys
import os
from importlib import import_module
import subprocess
import time
from . import tools

log = logging.getLogger(__name__)
class CrabHelper(object):

    def __init__(self):
        # perform imports only when creating instance. This allows to use the classmethods e.g.for
        # CLI construction before crab is sourced.
        self.crabFunctions =  import_module('CalibMuon.DTCalibration.Workflow.Crabtools.crabFunctions')
        # cached member variables
        self._crab = None
        self._cert_info = None

    def submit_crab_task(self):
        # create a crab config
        log.info("Creating crab config")
        self.create_crab_config()
        #write crab config
        full_crab_config_filename = self.write_crabConfig()
        if self.options.no_exec:
            log.info("Runing with option no-exec exiting")
            return True
        #submit crab job
        log.info("Submitting crab job")
        self.crab.submit(full_crab_config_filename)
        log.info("crab job submitted. Waiting 120 seconds before first status call")
        time.sleep( 120 )

        task = self.crabFunctions.CrabTask(crab_config = full_crab_config_filename)
        task.update()
        if task.state =="UNKNOWN":
            time.sleep( 30 )
            task.update()
        success_states = ( 'QUEUED', 'SUBMITTED', "COMPLETED", "FINISHED")
        if task.state in success_states:
            log.info("Job in state %s" % task.state )
            return True
        else:
            log.error("Job submission not successful, crab state:%s" % task.state)
            raise RuntimeError("Job submission not successful, crab state:%s" % task.state)

    def check_crabtask(self):
        print(self.crab_config_filepath)
        task = self.crabFunctions.CrabTask(crab_config = self.crab_config_filepath,
                                            initUpdate = False)
        if self.options.no_exec:
            log.info("Nothing to check in no-exec mode")
            return True
        for n_check in range(self.options.max_checks):
            task.update()
            if task.state in ( "COMPLETED"):
                print("Crab task complete. Getting output locally")
                output_path = os.path.join( self.local_path, "unmerged_results" )
                self.get_output_files(task, output_path)
                return True
            if task.state in ("SUBMITFAILED", "FAILED"):
                print("Crab task failed")
                return False
            possible_job_states =  ["nUnsubmitted",
                                    "nIdle",
                                    "nRunning",
                                    "nTransferring",
                                    "nCooloff",
                                    "nFailed",
                                    "nFinished",
                                    "nComplete" ]

            jobinfos = ""
            for jobstate in possible_job_states:
                njobs_in_state = getattr(task, jobstate)
                if njobs_in_state > 0:
                    jobinfos+="%s: %d " % (jobstate[1:], njobs_in_state)

            #clear line for reuse
            sys.stdout.write("\r")
            sys.stdout.write("".join([" " for i in range(tools.getTerminalSize()[0])]))
            sys.stdout.write("\r")
            prompt_text = "Check (%d/%d). Task state: %s (%s). Press q and enter to stop checks: " % (n_check,
                self.options.max_checks, task.state, jobinfos)
            user_input = tools.stdinWait(prompt_text, "", self.options.check_interval)
            if user_input in ("q","Q"):
                return False
        print("Task not completed after %d checks (%d minutes)" % ( self.options.max_checks,
            int( self.options.check_interval / 60. )))
        return False

    def voms_proxy_time_left(self):
        process = subprocess.Popen(['voms-proxy-info', '-timeleft'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        stdout = process.communicate()[0]
        if process.returncode != 0:
            return 0
        else:
            return int(stdout)

    def voms_proxy_create(self, passphrase = None):
        voms = 'cms'
        if passphrase:
            p = subprocess.Popen(['voms-proxy-init', '--voms', voms, '--valid', '192:00'],
                                 stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout = p.communicate(input=passphrase+'\n')[0]
            retcode = p.returncode
            if not retcode == 0:
                raise ProxyError('Proxy initialization command failed: %s'%stdout)
        else:
            retcode = subprocess.call(['voms-proxy-init', '--voms', voms, '--valid', '192:00'])
        if not retcode == 0:
            raise ProxyError('Proxy initialization command failed.')


    def create_crab_config(self):
        """ Create a crab config object dependent on the chosen command option"""
        from CalibMuon.DTCalibration.Workflow.Crabtools.crabConfigParser import CrabConfigParser
        self.crab_config = CrabConfigParser()
        """ Fill common options in crab config """
        ### General section
        self.crab_config.add_section('General')
        if "/" in self.crab_taskname:
            raise ValueError( 'Sample contains "/" which is not allowed' )
        self.crab_config.set( 'General', 'requestName', self.crab_taskname )
        self.crab_config.set( 'General', 'workArea', self.local_path)
        if self.options.no_log:
            self.crab_config.set( 'General', 'transferLogs', 'False' )
        else:
            self.crab_config.set( 'General', 'transferLogs', 'True' )
        ### JobType section
        self.crab_config.add_section('JobType')
        self.crab_config.set( 'JobType', 'pluginName', 'Analysis' )
        self.crab_config.set( 'JobType', 'psetName', self.pset_path )
        self.crab_config.set( 'JobType', 'outputFiles', self.output_files)
        if self.input_files:
            self.crab_config.set( 'JobType', 'inputFiles', self.input_files)
        ### Data section
        self.crab_config.add_section('Data')
        self.crab_config.set('Data', 'inputDataset', self.options.datasetpath)
        # set job splitting options
        if self.options.datasettype =="MC":
            self.crab_config.set('Data', 'splitting', 'FileBased')
            self.crab_config.set('Data', 'unitsPerJob', str(self.options.filesPerJob) )
        else:
            self.crab_config.set('Data', 'splitting', 'LumiBased')
            self.crab_config.set('Data', 'unitsPerJob', str(self.options.lumisPerJob) )
            if self.options.runselection:
                self.crab_config.set( "Data",
                                      "runRange",
                                      ",".join( self.options.runselection )
                                    )
        # set output path in compliance with crab3 structure
        self.crab_config.set('Data', 'publication', False)
        self.crab_config.set('Data', 'outputDatasetTag', self.remote_out_path["outputDatasetTag"])
        self.crab_config.set('Data', 'outLFNDirBase', self.remote_out_path["outLFNDirBase"] )

        # set site section options
        self.crab_config.add_section('Site')
        self.crab_config.set('Site', 'storageSite', self.options.output_site)
        self.crab_config.set('Site', 'whitelist', self.options.ce_white_list)
        self.crab_config.set('Site', 'blacklist', self.options.ce_black_list)

        #set user section options if necessary
#        if self.cert_info.voGroup or self.cert_info.voRole:
#            self.crab_config.add_section('User')
#            if self.cert_info.voGroup:
#                self.crab_config.set('User', "voGroup", self.cert_info.voGroup)
#            if self.cert_info.voRole:
#                self.crab_config.set('User', "voRole", self.cert_info.voRole)
        log.debug("Created crab config: %s " % self.crab_config_filename)

    def write_crabConfig(self):
        """ Write crab config file in working dir with label option as name """
        base_path = os.path.join( self.options.working_dir,self.local_path)
        filename = os.path.join( base_path, self.crab_config_filename)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if os.path.exists(filename):
            raise IOError("file %s alrady exits"%(filename))
        self.crab_config.writeCrabConfig(filename)
        log.info( 'created crab config file %s'%filename )
        return filename

    def fill_options_from_crab_config(self):
        crabtask = CrabTask( crab_config = self.crab_config_filename )
        splitinfo = crabtask.crabConfig.Data.outputDatasetTag.split("_")
        run, trial = splitinfo[0].split("Run")[-1], splitinfo[1].split("v")[-1]
        if not self.options.run:
            self.options.run = int(run)
        self.options.trail = int(trial)
        if not hasattr(self.options, "datasetpath"):
            self.options.datasetpath = crabtask.crabConfig.Data.inputDataset
        if not hasattr(self.options, "label"):
            self.options.label = crabtask.crabConfig.General.requestName.split("_")[0]

    @property
    def crab(self):
        """ Retuns a CrabController instance from cache or creates new
           on on first call """
        if self._crab is None:
            if self.cert_info.voGroup:
                self._crab = self.crabFunctions.CrabController(voGroup = self.cert_info.voGroup)
            else:
                self._crab = self.crabFunctions.CrabController()
        return self._crab

    @property
    def cert_info(self):
        if not self._cert_info:
            if not self.voms_proxy_time_left() > 0:
                warn_msg = "No valid proxy, a default proxy without a specific"
                warn_msg = "VOGroup will be used"
                self.voms_proxy_create()
                log.warning(warn_msg)
            self._cert_info = self.crabFunctions.CertInfo()
        return self._cert_info

    @property
    def crab_config_filename(self):
        if hasattr(self.options, "crab_config_path"):
            return self.options.crab_config_path
        return 'crab_%s_cfg.py' % self.crab_taskname

    @property
    def crab_config_filepath(self):
        base_path = os.path.join( self.options.working_dir,self.local_path)
        return os.path.join( base_path, self.crab_config_filename)

    @property
    def crab_taskname(self):
        taskname = self.options.label + "_" + self.options.workflow + "_"
        if hasattr( self.options, "workflow_mode"):
            taskname+= self.options.workflow_mode + "_"
        taskname += "run_" + str(self.options.run) + "_v" + str(self.options.trial)
        return taskname

## Exception for the VOMS proxy
class ProxyError(Exception):
    pass
