from __future__ import absolute_import
import os
import logging

from . import tools
import FWCore.ParameterSet.Config as cms
from .DTWorkflow import DTWorkflow

log = logging.getLogger(__name__)

class DTvdriftWorkflow( DTWorkflow ):
    """ This class creates and performce / submits vdrift workflow jobs"""
    def __init__(self, options):
        # call parent constructor
        super( DTvdriftWorkflow, self ).__init__( options )

        self.outpath_command_tag = "VdriftCalibration"
        output_file_dict = { "segment" : "DTVDriftHistos.root",
                             "meantimer" : "DTTMaxHistos.root",
                            }
        self.outpath_workflow_mode_dict = { "segment" : "Segments",
                                            "meantimer" : "MeanTimer",
                                            }
        self.output_file = output_file_dict[self.options.workflow_mode]
        self.output_files = [self.output_file]

    def prepare_workflow(self):
        """ Generalized function to prepare workflow dependent on workflow mode"""
        function_name = "prepare_" + self.options.workflow_mode + "_" + self.options.command

        try:
            fill_function = getattr(self, function_name)
        except AttributeError:
            errmsg = "Class `{}` does not implement `{}`"
            raise NotImplementedError( errmsg.format(self.__class__.__name__,
                                                     function_name))
        log.debug("Preparing workflow with function %s" % function_name)
        # call chosen function
        fill_function()

    def prepare_segment_submit(self):
        self.pset_name = 'dtVDriftSegmentCalibration_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtVDriftSegmentCalibration_cfg'
        if self.options.datasettype == "Cosmics":
            self.pset_template = 'CalibMuon.DTCalibration.dtVDriftSegmentCalibration_cosmics_cfg'

        self.process = tools.loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.options.globaltag
        self.process.dtVDriftSegmentCalibration.rootFileName = self.output_file

        if self.options.inputCalibDB:
            err = "Option inputCalibDB not available for segment."
            err += "Maybe you want to use option inputTtrigDB"
            raise ValueError(err)
        self.prepare_common_submit()
        if self.options.inputTtrigDB:
            label = ''
            if self.options.datasettype == "Cosmics":
                label = 'cosmics'
            connect = os.path.basename(self.options.inputTtrigDB)
            self.addPoolDBESSource( process = self.process,
                                    moduleName = 'tTrigDB',
                                    record = 'DTTtrigRcd',
                                    tag = 'ttrig',
                                    connect = str("sqlite_file:%s" % connect),
                                    label = label
                                    )
            self.input_files.append( os.path.abspath(self.options.inputTtrigDB) )
        self.write_pset_file()

    def prepare_segment_check(self):
        self.load_options_command("submit")

    def prepare_segment_write(self):
        self.pset_name = 'dtVDriftSegmentWriter_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtVDriftSegmentWriter_cfg'
        tag = self.prepare_common_write()
        merged_file = os.path.join(self.result_path, self.output_file)
        self.process = tools.loadCmsProcess(self.pset_template)

        if self.options.inputVDriftDB:
            self.add_local_vdrift_db(self)
        vdrift_db = "vDrift_segment"+ tag + ".db"
        vdrift_db = os.path.join(self.result_path, vdrift_db)
        self.process.dtVDriftSegmentWriter.vDriftAlgoConfig.rootFileName = "file:///" + merged_file
        self.process.PoolDBOutputService.connect = 'sqlite_file:%s' % vdrift_db
        self.process.source.firstRun = cms.untracked.uint32(self.options.run)
        self.process.GlobalTag.globaltag = cms.string(str(self.options.globaltag))
        self.write_pset_file()

    def prepare_segment_dump(self):
        self.pset_name = 'dumpDBToFile_vdrift_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dumpDBToFile_vdrift_cfg'
        if self.options.input_dumpDB:
            try:
                test = self.result_path
                self.load_options_command("write")
            except:
                pass
            dbpath = os.path.abspath(self.options.input_dumpDB)
        else:
            crabtask = self.crabFunctions.CrabTask(crab_config = self.crab_config_filepath,
                                               initUpdate = False)
            tag = crabtask.crabConfig.Data.outputDatasetTag
            dbpath = os.path.abspath( os.path.join(self.result_path,
                                                   "vDrift_segment"+ tag + ".db"))
        self.prepare_common_dump(dbpath)
        self.write_pset_file()

    def prepare_segment_all(self):
        # individual prepare functions for all tasks will be called in
        # main implementation of all
        self.all_commands=["submit", "check", "write", "dump"]

    ####################################################################
    #                            Mean Timer                            #
    ####################################################################
    def prepare_meantimer_submit(self):
        self.pset_name = 'dtVDriftMeanTimerCalibration_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtVDriftMeanTimerCalibration_cfg'
        if self.options.datasettype == "Cosmics":
            self.pset_template = 'CalibMuon.DTCalibration.dtVDriftMeanTimerCalibration_cosmics_cfg'

        self.process = tools.loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.options.globaltag
        self.process.dtVDriftMeanTimerCalibration.rootFileName = self.output_file

        if self.options.inputCalibDB:
            err = "Option inputCalibDB not available for meantimer."
            err += "Maybe you want to use option inputTtrigDB"
            raise ValueError(err)
        self.prepare_common_submit()
        if self.options.inputTtrigDB:
            label = ''
            if self.options.datasettype == "Cosmics":
                label = 'cosmics'
            connect = os.path.basename(self.options.inputTtrigDB)
            self.addPoolDBESSource( process = self.process,
                                    moduleName = 'tTrigDB',
                                    record = 'DTTtrigRcd',
                                    tag = 'ttrig',
                                    connect = str("sqlite_file:%s" % connect),
                                    label = label
                                    )
            self.input_files.append( os.path.abspath(self.options.inputTtrigDB) )
        self.write_pset_file()

    def prepare_meantimer_check(self):
        self.load_options_command("submit")

    def prepare_meantimer_write(self):
        self.pset_name = 'dtVDriftMeanTimerWriter_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtVDriftMeanTimerWriter_cfg'
        tag = self.prepare_common_write()
        merged_file = os.path.join(self.result_path, self.output_file)
        self.process = tools.loadCmsProcess(self.pset_template)

        if self.options.inputVDriftDB:
            self.add_local_vdrift_db(self)
        vdrift_db = "vDrift_meantimer" + tag + ".db"
        vdrift_db = os.path.join(self.result_path, vdrift_db)
        self.process.dtVDriftMeanTimerWriter.vDriftAlgoConfig.rootFileName = "file:///" + merged_file
        self.process.PoolDBOutputService.connect = 'sqlite_file:%s' % vdrift_db
        self.process.source.firstRun = cms.untracked.uint32(self.options.run)
        self.process.GlobalTag.globaltag = cms.string(str(self.options.globaltag))
        self.write_pset_file()

    def prepare_meantimer_dump(self):
        self.pset_name = 'dumpDBToFile_vdrift_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dumpDBToFile_vdrift_cfg'
        if self.options.input_dumpDB:
            try:
                test = self.result_path
                self.load_options_command("write")
            except:
                pass
            dbpath = os.path.abspath(self.options.input_dumpDB)
        else:
            crabtask = self.crabFunctions.CrabTask(crab_config = self.crab_config_filepath,
                                               initUpdate = False)
            tag = crabtask.crabConfig.Data.outputDatasetTag
            dbpath = os.path.abspath( os.path.join(self.result_path,
                                                   "vDrift_meantimer" + tag + ".db"))
        self.prepare_common_dump(dbpath)
        self.write_pset_file()

    def prepare_meantimer_all(self):
        # individual prepare functions for all tasks will be called in
        # main implementation of all
        self.all_commands=["submit", "check", "write", "dump"]

    ####################################################################
    #                           CLI creation                           #
    ####################################################################
    @classmethod
    def add_parser_options(cls, subparser_container):
        vdrift_parser = subparser_container.add_parser( "vdrift",
        #parents=[mutual_parent_parser, common_parent_parser],
        help = "" ) # What does ttrig

        ################################################################
        #                Sub parser options for workflow modes         #
        ################################################################
        vdrift_subparsers = vdrift_parser.add_subparsers( dest="workflow_mode",
            help="Possible workflow modes",)
        ## Add all workflow modes for ttrig
        vdrift_segment_subparser = vdrift_subparsers.add_parser( "segment",
            #parents=[mutual_parent_parser, common_parent_parser],
            help = "" )
        vdrift_meantimer_subparser = vdrift_subparsers.add_parser( "meantimer",
            #parents=[mutual_parent_parser, common_parent_parser],
            help = "" )
        ################################################################
        #        Sub parser options for workflow mode segment          #
        ################################################################
        vdrift_segment_subparsers = vdrift_segment_subparser.add_subparsers( dest="command",
            help="Possible commands for segments")
        vdrift_segment_submit_parser = vdrift_segment_subparsers.add_parser(
            "submit",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                    super(DTvdriftWorkflow,cls).get_submission_options_parser(),
                    super(DTvdriftWorkflow,cls).get_local_input_db_options_parser(),
                    super(DTvdriftWorkflow,cls).get_input_db_options_parser()],
            help = "Submit job to the GRID via crab3")
        vdrift_segment_submit_parser.add_argument("--inputTtrigDB",
            help="Local alternative calib ttrig db")

        vdrift_segment_check_parser = vdrift_segment_subparsers.add_parser(
            "check",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                    super(DTvdriftWorkflow,cls).get_check_options_parser()],
            help = "Check status of submitted jobs")

        vdrift_segment_write_parser = vdrift_segment_subparsers.add_parser(
            "write",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                     super(DTvdriftWorkflow,cls).get_write_options_parser()
                    ],
            help = "Write result from root output to text file")

        vdrift_segment_dump_parser = vdrift_segment_subparsers.add_parser(
            "dump",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                     super(DTvdriftWorkflow,cls).get_dump_options_parser()],
            help = "Dump database to text file")

        vdrift_segment_all_parser = vdrift_segment_subparsers.add_parser(
            "all",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                     super(DTvdriftWorkflow,cls).get_submission_options_parser(),
                     super(DTvdriftWorkflow,cls).get_check_options_parser(),
                     super(DTvdriftWorkflow,cls).get_input_db_options_parser(),
                     super(DTvdriftWorkflow,cls).get_local_input_db_options_parser(),
                     super(DTvdriftWorkflow,cls).get_write_options_parser(),
                     super(DTvdriftWorkflow,cls).get_dump_options_parser()
                    ],
            help = "Perform all steps: submit, check, write, dump in this order")
        vdrift_segment_all_parser.add_argument("--inputTtrigDB",
            help="Local alternative calib ttrig db")
        ################################################################
        #       Sub parser options for workflow mode meantimer         #
        ################################################################
        vdrift_meantimer_subparsers = vdrift_meantimer_subparser.add_subparsers( dest="command",
            help="Possible commands for meantimers")
        vdrift_meantimer_submit_parser = vdrift_meantimer_subparsers.add_parser(
            "submit",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                    super(DTvdriftWorkflow,cls).get_submission_options_parser(),
                    super(DTvdriftWorkflow,cls).get_local_input_db_options_parser(),
                    super(DTvdriftWorkflow,cls).get_input_db_options_parser()],
            help = "Submit job to the GRID via crab3")
        vdrift_meantimer_submit_parser.add_argument("--inputTtrigDB",
            help="Local alternative calib ttrig db")

        vdrift_meantimer_check_parser = vdrift_meantimer_subparsers.add_parser(
            "check",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                    super(DTvdriftWorkflow,cls).get_check_options_parser()],
            help = "Check status of submitted jobs")

        vdrift_meantimer_write_parser = vdrift_meantimer_subparsers.add_parser(
            "write",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                     super(DTvdriftWorkflow,cls).get_write_options_parser()
                    ],
            help = "Write result from root output to text file")

        vdrift_meantimer_dump_parser = vdrift_meantimer_subparsers.add_parser(
            "dump",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                     super(DTvdriftWorkflow,cls).get_dump_options_parser()],
            help = "Dump database to text file")

        vdrift_meantimer_all_parser = vdrift_meantimer_subparsers.add_parser(
            "all",
            parents=[super(DTvdriftWorkflow,cls).get_common_options_parser(),
                     super(DTvdriftWorkflow,cls).get_submission_options_parser(),
                     super(DTvdriftWorkflow,cls).get_check_options_parser(),
                     super(DTvdriftWorkflow,cls).get_input_db_options_parser(),
                     super(DTvdriftWorkflow,cls).get_local_input_db_options_parser(),
                     super(DTvdriftWorkflow,cls).get_write_options_parser(),
                     super(DTvdriftWorkflow,cls).get_dump_options_parser()
                    ],
            help = "Perform all steps: submit, check, write, dump in this order")
        vdrift_meantimer_all_parser.add_argument("--inputTtrigDB",
            help="Local alternative calib ttrig db")
