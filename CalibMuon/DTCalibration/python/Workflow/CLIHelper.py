import os
import argparse
class CLIHelper(object):
    @classmethod
    def get_common_options_parser(cls):
        """ Return a parser with common options for each workflow"""
        common_opts_parser = argparse.ArgumentParser(add_help=False)
        common_opts_group = common_opts_parser.add_argument_group(
            description ="General options")
        common_opts_group.add_argument("-r","--run", type=int,
            help="set reference run number (typically first or last run in list)")
        common_opts_group.add_argument("--trial", type=int, default = 1,
            help="trial number used in the naming of output directories")
        common_opts_group.add_argument("--label", default="dtCalibration",
            help="label used in the naming of workflow output default:%(default)s")
        common_opts_group.add_argument("--datasettype", default = "Data",
            choices=["Data", "Cosmics", "MC"], help="Type of input dataset default: %(default)s")
        common_opts_group.add_argument("--config-path",
            help="Path to alternative workflow config json file, e.g. used to submit the job")
        common_opts_group.add_argument("--user", default="",
            help="User used e.g. for submission. Defaults to user HN name")
        common_opts_group.add_argument("--working-dir",
            default=os.getcwd(), help="connect string default:%(default)s")
        common_opts_group.add_argument("--no-exec",
            action="store_true", help="Do not execute or submit any workflow")
        return common_opts_parser

    @classmethod
    def add_parser_options(cls, parser):
        # Subparsers are used to choose a calibration workflow
        workflow_subparsers = parser.add_subparsers( help="workflow option help", dest="workflow" )
        return workflow_subparsers


    def fill_required_options_prepare_dict(self):
        common_required = []
        self.required_options_prepare_dict["submit"] = ["globaltag"]

    def fill_required_options_dict(self):
        common_required = ["run"]
        self.required_options_dict["submit"] = common_required
        self.required_options_dict["submit"].append("datasetpath")
        self.required_options_dict["submit"].append("globaltag")

        self.required_options_dict["correction"] = common_required
        self.required_options_dict["correction"].append("globaltag")


    @classmethod
    def get_input_db_options_parser(cls):
        """ Return a parser object with options relevant for input databases"""
        db_opts_parser = argparse.ArgumentParser(add_help=False)
        dp_opts_group = db_opts_parser.add_argument_group(
            description ="Options for Input databases")
        db_opts_parser.add_argument("--inputDBRcd",
            help="Record used for PoolDBESSource")
        db_opts_parser.add_argument("--inputDBTag",
            help="Tag used for PoolDBESSource")
        db_opts_parser.add_argument("--connectStrDBTag",
            default='frontier://FrontierProd/CMS_COND_31X_DT',
            help="connect string default:%(default)s")
        return db_opts_parser

    @classmethod
    def get_local_input_db_options_parser(cls):
        """ Return a parser object with options relevant for input databases"""
        db_opts_parser = argparse.ArgumentParser(add_help=False)
        db_opts_group = db_opts_parser.add_argument_group(
            description ="Options for local input databases")
        db_opts_group.add_argument("--inputVDriftDB",
            help="Local alternative VDrift database")
        db_opts_group.add_argument("--inputCalibDB",
            help="Local alternative Ttrig database")
        db_opts_group.add_argument("--inputT0DB",
            help="Local alternative T0 database")
        return db_opts_parser

    @classmethod
    def get_submission_options_parser(cls):
        """ Return a parser object with options relevant to remote submission"""
        submission_opts_parser = argparse.ArgumentParser(add_help=False)
        submission_opts_group = submission_opts_parser.add_argument_group(
            description ="Options for Job submission")
        submission_opts_group.add_argument("--datasetpath",
            help="dataset name to process")
        submission_opts_group.add_argument("--run-on-RAW", action = "store_true",
            help="Flag if run on RAW dataset")
        submission_opts_group.add_argument("--fromMuons", action = "store_true",
            help="Segment selection using muon-segment matching")
        submission_opts_group.add_argument("--globaltag",
            help="global tag identifier (with the '::All' string, if necessary)")
        submission_opts_group.add_argument("--histoRange", default = 0.4,
            help="Range or residual histogram, default is 0.4cm")
        submission_opts_group.add_argument("--runselection", default = [], nargs="+",
            help="run list or range")
        submission_opts_group.add_argument("--filesPerJob", default = 5,
            help="Number of files to process for MC grid jobs")
        submission_opts_group.add_argument("--lumisPerJob", default = 10000,
            help="Number of lumi sections to process for RAW / Comsics grid jobs")
        submission_opts_group.add_argument("--preselection", dest="preselection",
            help="configuration fragment and sequence name, separated by a ':', defining a pre-selection filter")
        submission_opts_group.add_argument("--output-site", default = "T2_DE_RWTH",
            help="Site used for stage out of results")
        submission_opts_group.add_argument("--ce-black-list", default = [], nargs="+",
            help="add sites to black list when run on Grid")
        submission_opts_group.add_argument("--ce-white-list", default = [], nargs="+",
            help="add sites to white list when run on Grid")
        submission_opts_group.add_argument("--no-log",
            action="store_true", help="Do not transfer crab logs:%(default)s")
        return submission_opts_parser

    @classmethod
    def get_check_options_parser(cls):
        """ Return a parser object with options relevant to check the status of remote submission"""
        check_opts_parser = argparse.ArgumentParser(add_help=False)
        check_opts_group = check_opts_parser.add_argument_group(
            description ="Options for Job submission")
        check_opts_group.add_argument("--check-interval", default = 600,type=int,
            help="Time in seconds between check operations default: %(default)s")
        check_opts_group.add_argument("--max-checks", default =1000, type=int,
            help="Maximum number of checks before check is considered failed default: %(default)s")
        return check_opts_parser

    @classmethod
    def get_write_options_parser(cls):
        """ Return a parser object with options relevant to write results to dbs"""
        check_opts_parser = argparse.ArgumentParser(add_help=False)
        check_opts_group = check_opts_parser.add_argument_group(
            description ="Options for write jobs")
        check_opts_group.add_argument("--skip-stageout", action="store_true",
            help="Skip stageout to local disk and merging")
        return check_opts_parser

    @classmethod
    def get_dump_options_parser(cls):
        dump_opts_parser = argparse.ArgumentParser(add_help=False)
        dump_opts_group = dump_opts_parser.add_argument_group(
            description ="Options for dump db file")
        dump_opts_group.add_argument("--input-dumpDB",
        help="Input database file to dump."\
             " Defaults to existing corrected database from correction command"\
             " if run, label, trial or input config are specified")
        return dump_opts_group

