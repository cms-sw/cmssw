import os,sys
import glob
import logging
import argparse
import subprocess
import time, datetime
import urllib2
import json

import tools
from CLIHelper import CLIHelper
from CrabHelper import CrabHelper
import FWCore.ParameterSet.Config as cms
log = logging.getLogger(__name__)

class DTWorkflow(CLIHelper, CrabHelper):
    """ This is the base class for all DTWorkflows and contains some
        common tasks """
    def __init__(self, options):
        self.options = options
        super( DTWorkflow, self ).__init__()
        self.digilabel = "muonDTDigis"
        # dict to hold required variables. Can not be marked in argparse to allow
        # loading of options from config
        self.required_options_dict = {}
        self.required_options_prepare_dict = {}
        self.fill_required_options_dict()
        self.fill_required_options_prepare_dict()
        # These variables are determined in the derived classes
        self.pset_name = ""
        self.outpath_command_tag = ""
        self.output_files = []
        self.input_files = []

        self.run_all_command = False
        self.files_reveived = False
        self._user = ""
        # change to working directory
        os.chdir(self.options.working_dir)

    def check_missing_options(self, requirements_dict):
        missing_options = []
        # check if all required options exist
        if self.options.command in requirements_dict:
            for option in requirements_dict[self.options.command]:
                if not (hasattr(self.options, option)
                    and ( (getattr(self.options,option))
                          or isinstance(getattr(self.options,option), bool) )):
                    missing_options.append(option)
        if len(missing_options) > 0:
            err = "The following CLI options are missing"
            err += " for command %s: " % self.options.command
            err += " ".join(missing_options)
            raise ValueError(err)

    def run(self):
        """ Generalized function to run workflow command"""
        msg = "Preparing %s workflow" % self.options.workflow
        if hasattr(self.options, "command"):
            msg += " for command %s" % self.options.command
        log.info(msg)
        if self.options.config_path:
            self.load_options( self.options.config_path )
        #check if all options to prepare the command are used
        self.check_missing_options(self.required_options_prepare_dict)
        self.prepare_workflow()
        # create output folder if they do not exist yet
        if not os.path.exists( self.local_path ):
            os.makedirs(self.local_path)
        # dump used options
        self.dump_options()
        #check if all options to run the command are used
        self.check_missing_options(self.required_options_dict)
        try:
            run_function = getattr(self, self.options.command)
        except AttributeError:
            errmsg = "Class `{}` does not implement `{}` for workflow %s" % self.options.workflow
            if hasattr(self.options, "workflow_mode"):
                errmsg += "and workflow mode %s" % self.options.workflow_mode
            raise NotImplementedError( errmsg.format(self.__class__.__name__,
                                                     self.options.command))
        log.debug("Running command %s" % self.options.command)
        # call chosen function
        run_function()

    def prepare_workflow(self):
        """ Abstract implementation of prepare workflow function"""
        errmsg = "Class `{}` does not implement `{}`"
        raise NotImplementedError( errmsg.format(self.__class__.__name__,
                                                     "prepare_workflow"))

    def all(self):
        """ generalized function to perform several workflow mode commands in chain.
            All commands mus be specified in self.all_commands list in workflow mode specific
            prepare function in child workflow objects.
        """
        self.run_all_command = True
        for command in self.all_commands:
            self.options.command = command
            self.run()

    def submit(self):
        self.submit_crab_task()

    def check(self):
        """ Function to check status of submitted tasks """
        self.check_crabtask()

    def write(self):
        self.runCMSSWtask()

    def dump(self):
        self.runCMSSWtask()

    def correction(self):
        self.runCMSSWtask()

    def add_preselection(self):
        """ Add preselection to the process object stored in workflow_object"""
        if not hasattr(self, "process"):
            raise NameError("Process is not initalized in workflow object")
        pathsequence = self.options.preselection.split(':')[0]
        seqname = self.options.preselection.split(':')[1]
        self.process.load(pathsequence)
        tools.prependPaths(self.process, seqname)

    def add_raw_option(self):
        getattr(self.process, self.digilabel).inputLabel = 'rawDataCollector'
        tools.prependPaths(self.process,self.digilabel)

    def add_local_t0_db(self, local=False):
        """ Add a local t0 database as input. Use the option local is used
            if the pset is processed locally and not with crab.
        """
        if local:
            connect = os.path.abspath(self.options.inputT0DB)
        else:
            connect = os.path.basename(self.options.inputT0DB)
        self.addPoolDBESSource( process = self.process,
                                moduleName = 't0DB',
                                record = 'DTT0Rcd',
                                tag = 't0',
                                connect =  'sqlite_file:%s' % connect)
        self.input_files.append(os.path.abspath(self.options.inputT0DB))

    def add_local_vdrift_db(self, local=False):
        """ Add a local vdrift database as input. Use the option local is used
            if the pset is processed locally and not with crab.
         """
        if local:
            connect = os.path.abspath(self.options.inputVDriftDB)
        else:
            connect = os.path.basename(self.options.inputVDriftDB)
        self.addPoolDBESSource( process = self.process,
                                moduleName = 'vDriftDB',
                                record = 'DTMtimeRcd',
                                tag = 'vDrift',
                                connect = 'sqlite_file:%s' % connect)
        self.input_files.append( os.path.abspath(self.options.inputVDriftDB) )

    def add_local_calib_db(self, local=False):
        """ Add a local calib database as input. Use the option local is used
            if the pset is processed locally and not with crab.
         """
        label = ''
        if self.options.datasettype == "Cosmics":
            label = 'cosmics'
        if local:
            connect = os.path.abspath(self.options.inputCalibDB)
        else:
            connect = os.path.basename(self.options.inputCalibDB)
        self.addPoolDBESSource( process = self.process,
                                moduleName = 'calibDB',
                                record = 'DTTtrigRcd',
                                tag = 'ttrig',
                                connect = str("sqlite_file:%s" % connect),
                                label = label
                                )
        self.input_files.append( os.path.abspath(self.options.inputCalibDB) )

    def add_local_custom_db(self):
        for option in ('inputDBRcd', 'connectStrDBTag'):
            if hasattr(self.options, option) and not getattr(self.options, option):
                raise ValueError("Option %s needed for custom input db" % option)
        self.addPoolDBESSource( process = self.process,
                                    record = self.options.inputDBRcd,
                                    tag = self.options.inputDBTag,
                                    connect = self.options.connectStrDBTag,
                                    moduleName = 'customDB%s' % self.options.inputDBRcd
                                   )

    def prepare_common_submit(self):
        """ Common operations used in most prepare_[workflow_mode]_submit functions"""
        if not self.options.run:
            raise ValueError("Option run is required for submission!")
        if hasattr(self.options, "inputT0DB") and self.options.inputT0DB:
            self.add_local_t0_db()

        if hasattr(self.options, "inputVDriftDB") and self.options.inputVDriftDB:
            self.add_local_vdrift_db()

        if hasattr(self.options, "inputDBTag") and self.options.inputDBTag:
            self.add_local_custom_db()

        if self.options.run_on_RAW:
            self.add_raw_option()
        if self.options.preselection:
            self.add_preselection()

    def prepare_common_write(self, do_hadd=True):
        """ Common operations used in most prepare_[workflow_mode]_erite functions"""
        self.load_options_command("submit")
        output_path = os.path.join( self.local_path, "unmerged_results" )
        merged_file = os.path.join(self.result_path, self.output_file)
        crabtask = self.crabFunctions.CrabTask(crab_config = self.crab_config_filepath,
                                               initUpdate = False)
        if not (self.options.skip_stageout or self.files_reveived or self.options.no_exec):
            self.get_output_files(crabtask, output_path)
            log.info("Received files from storage element")
            log.info("Using hadd to merge output files")
        if not self.options.no_exec and do_hadd:
            returncode = tools.haddLocal(output_path, merged_file)
            if returncode != 0:
                raise RuntimeError("Failed to merge files with hadd")
        return crabtask.crabConfig.Data.outputDatasetTag

    def prepare_common_dump(self, db_path):
        self.process = tools.loadCmsProcess(self.pset_template)
        self.process.calibDB.connect = 'sqlite_file:%s' % db_path
        try:
            path = self.result_path
        except:
            path = os.getcwd()
        print "path", path
        out_path = os.path.abspath(os.path.join(path,
                                                os.path.splitext(db_path)[0] + ".txt"))

        self.process.dumpToFile.outputFileName = out_path

    @staticmethod
    def addPoolDBESSource( process,
                           moduleName,
                           record,
                           tag,
                           connect='sqlite_file:',
                           label='',):

        from CondCore.CondDB.CondDB_cfi import CondDB

        calibDB = cms.ESSource("PoolDBESSource",
                               CondDB,
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(cms.PSet(
                                   record = cms.string(record),
                                   tag = cms.string(tag),
                                   label = cms.untracked.string(label)
                                    )),
                               )
        calibDB.connect = cms.string( str(connect) )
        #if authPath: calibDB.DBParameters.authenticationPath = authPath
        if 'oracle:' in connect:
            calibDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
        setattr(process,moduleName,calibDB)
        setattr(process,"es_prefer_" + moduleName,cms.ESPrefer('PoolDBESSource',
                                                                moduleName)
                                                                )

    def get_output_files(self, crabtask, output_path):
        self.crab.callCrabCommand( ["getoutput",
                                    "--outputpath",
                                    output_path,
                                    crabtask.crabFolder ] )

    def runCMSSWtask(self, pset_path=""):
        """ Run a cmsRun job locally. The member variable self.pset_path is used
            if pset_path argument is not given"""
        if self.options.no_exec:
            return 0
        process = subprocess.Popen( "cmsRun %s" % self.pset_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell = True)
        stdout = process.communicate()[0]
        log.info(stdout)
        if process.returncode != 0:
            raise RuntimeError("Failed to use cmsRun for pset %s" % self.pset_name)
        return process.returncode

    @property
    def remote_out_path(self):
        """ Output path on remote excluding user base path
        Returns a dict if crab is used due to crab path setting policy"""
        if self.options.command =="submit":
            return {
                "outLFNDirBase" : os.path.join( "/store",
                                                "user",
                                                self.user,
                                                'DTCalibration/',
                                                self.outpath_command_tag,
                                                self.outpath_workflow_mode_tag),
                "outputDatasetTag" : self.tag
                    }
        else:
            return os.path.join( 'DTCalibration/',
                                 datasetstr,
                                 'Run' + str(self.options.run),
                                 self.outpath_command_tag,
                                 self.outpath_workflow_mode_tag,
                                 'v' + str(self.options.trial),
                                )
    @property
    def outpath_workflow_mode_tag(self):
        if not self.options.workflow_mode in self.outpath_workflow_mode_dict:
            raise NotImplementedError("%s missing in outpath_workflow_mode_dict" % self.options.workflow_mode)
        return self.outpath_workflow_mode_dict[self.options.workflow_mode]

    @property
    def tag(self):
        return 'Run' + str(self.options.run) + '_v' + str(self.options.trial)

    @property
    def user(self):
        if self._user:
            return self._user
        if hasattr(self.options, "user") and self.options.user:
            self._user = self.options.user
        else:
            self._user = self.crab.checkusername()
        return self._user

    @property
    def local_path(self):
        """ Output path on local machine """
        if self.options.run and self.options.label:
            prefix = "Run%d-%s_v%d" % ( self.options.run,
                                        self.options.label,
                                        self.options.trial)
        else:
            prefix = ""
        if self.outpath_workflow_mode_tag:
            path = os.path.join( self.options.working_dir,
                                 prefix,
                                 self.outpath_workflow_mode_tag)
        else:
            path =  os.path.join( self.options.working_dir,
                                  prefix,
                                  self.outpath_command_tag )
        return path

    @property
    def result_path(self):
        result_path = os.path.abspath(os.path.join(self.local_path,"results"))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        return result_path

    @property
    def pset_template_base_bath(self):
        """ Base path to folder containing pset files for cmsRun"""
        return os.path.expandvars(os.path.join("$CMSSW_BASE",
                                               "src",
                                               "CalibMuon",
                                               "test",
                                               )
                                 )

    @property
    def pset_path(self):
        """ full path to the pset file """
        basepath = os.path.join( self.local_path, "psets")
        if not os.path.exists( basepath ):
            os.makedirs( basepath )
        return os.path.join( basepath, self.pset_name )

    def write_pset_file(self):
        if not hasattr(self, "process"):
            raise NameError("Process is not initalized in workflow object")
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        with open( self.pset_path,'w') as pfile:
            pfile.write(self.process.dumpPython())

    def get_config_name(self, command= ""):
        """ Create the name for the output json file which will be dumped"""
        if not command:
            command = self.options.command
        return "config_" + command + ".json"

    def dump_options(self):
        with open(os.path.join(self.local_path, self.get_config_name()),"w") as out_file:
            json.dump(vars(self.options), out_file, indent=4)

    def load_options(self, config_file_path):
        if not os.path.exists(config_file_path):
            raise IOError("File %s not found" % config_file_path)
        with open(config_file_path, "r") as input_file:
            config_json = json.load(input_file)
            for key, val in config_json.items():
                if not hasattr(self.options, key) or not getattr(self.options, key):
                    setattr(self.options, key, val)

    def load_options_command(self, command ):
        """Load options for previous command in workflow """
        if not self.options.config_path:
            if not self.options.run:
                raise RuntimeError("Option run is required if no config path specified")
            if not os.path.exists(self.local_path):
                raise IOError("Local path %s does not exist" % self.local_path)
            self.options.config_path = os.path.join(self.local_path,
                                                    self.get_config_name(command))
        self.load_options( self.options.config_path )

