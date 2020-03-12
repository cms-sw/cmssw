import os
import logging

import tools
import FWCore.ParameterSet.Config as cms
from DTWorkflow import DTWorkflow

log = logging.getLogger(__name__)

class DTT0WireWorkflow(DTWorkflow):
    """ This class creates and performce / submits vdrift workflow jobs"""
    def __init__(self, options):
        # call parent constructor
        super(DTT0WireWorkflow, self).__init__(options)

        self.outpath_command_tag = "T0WireCalibration"
        self.outpath_workflow_mode_dict = {"all" : "All"}
        self.output_file = 'DTTestPulses.root'
        self.output_files = ['t0.db', self.output_file, 'DQM.root']

    def prepare_workflow(self):
        """ Generalized function to prepare workflow dependent on workflow mode"""
        function_name = "prepare_" + self.options.workflow_mode + "_" + self.options.command

        try:
            fill_function = getattr(self, function_name)
        except AttributeError:
            errmsg = "Class `{}` does not implement `{}`"
            raise NotImplementedError(errmsg.format(my_cls.__class__.__name__,
                                                    method_name))
        log.debug("Preparing workflow with function %s" % function_name)
        # call chosen function
        fill_function()

    def prepare_all_submit(self):
        self.pset_name = 'dtT0WireCalibration_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtT0WireCalibration_cfg'

        self.process = tools.loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.options.globaltag
        self.process.dtT0WireCalibration.rootFileName = self.output_file

        self.prepare_common_submit()
        self.write_pset_file()

    def prepare_all_all(self):
        # individual prepare functions for all tasks will be called in
        # main implementation of all
        self.all_commands=["submit"]

    def submit(self):
        # Overload to run locally
        self.runCMSSWtask()

    ####################################################################
    #                           CLI creation                           #
    ####################################################################
    @classmethod
    def add_parser_options(cls, subparser_container):
        vdrift_parser = subparser_container.add_parser( "T0Wire",
        #parents=[mutual_parent_parser, common_parent_parser],
        help = "" ) # What does ttrig

        ################################################################
        #                Sub parser options for workflow modes         #
        ################################################################
        vdrift_subparsers = vdrift_parser.add_subparsers( dest="workflow_mode",
            help="Possible workflow modes",)
        ## Add all workflow modes for ttrig
        vdrift_segment_subparser = vdrift_subparsers.add_parser( "all",
            #parents=[mutual_parent_parser, common_parent_parser],
            help = "" )
        ################################################################
        #        Sub parser options for workflow mode segment          #
        ################################################################
        vdrift_segment_subparsers = vdrift_segment_subparser.add_subparsers( dest="command",
            help="Possible commands for all")
        vdrift_segment_submit_parser = vdrift_segment_subparsers.add_parser(
            "submit",
            parents=[super(DTT0WireWorkflow,cls).get_common_options_parser(),
                    super(DTT0WireWorkflow,cls).get_submission_options_parser(),
                    super(DTT0WireWorkflow,cls).get_local_input_db_options_parser(),
                    super(DTT0WireWorkflow,cls).get_input_db_options_parser()],
            help = "Run job locally as GRID submission is not supported for T0 Calibration")

        vdrift_segment_all_parser = vdrift_segment_subparsers.add_parser(
            "all",
            parents=[super(DTT0WireWorkflow,cls).get_common_options_parser(),
                     super(DTT0WireWorkflow,cls).get_submission_options_parser(),
                     super(DTT0WireWorkflow,cls).get_input_db_options_parser(),
                     super(DTT0WireWorkflow,cls).get_local_input_db_options_parser(),
                    ],
            help = "Perform all steps: submit, check in this order")
