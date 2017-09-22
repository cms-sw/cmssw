#!/usr/bin/env python

#############################################################################
##  Builds the config-templates from the universal config-template for each
##  dataset specified in .ini-file that is passed to this script as argument.
##  Then calls mps_setup.pl for all datasets.
##
##  Usage:
##     mps_alisetup.py [-h] [-v] [-w] myconfig.ini
##

import argparse
import os
import re
import subprocess
import ConfigParser
import sys
import cPickle
import itertools
import collections
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.iniparser as mpsv_iniparser
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.trackerTree as mpsv_trackerTree
from Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper import checked_out_MPS
from functools import reduce


################################################################################
def main(argv = None):
    """Main routine. Not called, if this module is loaded via `import`.

    Arguments:
    - `argv`: Command line arguments passed to the script.
    """

    if argv == None:
        argv = sys.argv[1:]

    setup_alignment = SetupAlignment(argv)
    setup_alignment.setup()


################################################################################
class SetupAlignment(object):
    """Class encapsulating the alignment campaign setup procedure."""

    def __init__(self, argv):
        """Constructor

        Arguments:
        - `argv`: command line arguments
        """

        self._argv = argv       # raw command line arguments
        self._args = None       # parsed command line arguments
        self._config = None     # ConfigParser object
        self._mss_dir = None    # mass storage directory
        self._first_run = None   # first run for start geometry
        self._cms_process = None # cms.Process extracted from CMSSW config
        self._override_gt = None # snippet to append to config
        self._pede_script = None # path to pede batch script template
        self._mille_script = None # path to mille batch script template
        self._mps_dir_name = None # MP campaign name (mp<ID>)
        self._weight_configs = [] # list with combinations of dataset weights
        self._general_options = {} # general options extracted from ini file
        self._first_pede_config = True # does a pede job exist already?

        self._create_config()
        self._construct_paths()
        self._fill_general_options()
        self._create_mass_storage_directory()
        self._fetch_pede_settings()
        self._create_weight_configs()


    def setup(self):
        """Setup the alignment campaign."""

        if self._args.weight:
            self._create_additional_pede_jobs()
        else:
            self._create_mille_jobs()
            self._create_pede_jobs()

        if self._override_gt.strip() != "":
            print "="*60
            msg = ("Overriding global tag with single-IOV tags extracted from "
                   "'{}' for run number '{}'.".format(self._global_tag,
                                                      self._first_run))
            print msg
            print "-"*60
            print self._override_gt



    def _create_config(self):
        """Create ConfigParser object from command line arguments."""

        helpEpilog ="""Builds the config-templates from a universal
        config-template for each dataset specified in .ini-file that is passed
        to this script.  Then calls mps_setup.pl for all datasets."""
        parser = argparse.ArgumentParser(
            description = ("Setup the alignment as configured in the "
                           "alignment_config file."),
            epilog = helpEpilog)
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="display detailed output of mps_setup")
        parser.add_argument("-w", "--weight", action="store_true",
                            help=("creates additional merge job(s) with "
                                  "(possibly new) weights from .ini-config"))
        parser.add_argument("alignmentConfig",
                            help=("name of the .ini config file that specifies "
                                  "the datasets to be used"))

        self._args = parser.parse_args(self._argv)
        self._config = ConfigParser.ConfigParser()
        self._config.optionxform = str # default would give lowercase options
                                       # -> not wanted
        self._config.read(self._args.alignmentConfig)


    def _construct_paths(self):
        """Determine directory paths and create the ones that are needed."""

        mpsTemplates = os.path.join("src", "Alignment",
                                    "MillePedeAlignmentAlgorithm", "templates")
        if checked_out_MPS()[0]:
            mpsTemplates = os.path.join(os.environ["CMSSW_BASE"], mpsTemplates)
        else:
            mpsTemplates = os.path.join(os.environ["CMSSW_RELEASE_BASE"], mpsTemplates)
        self._mille_script = os.path.join(mpsTemplates, "mps_runMille_template.sh")
        self._pede_script  = os.path.join(mpsTemplates, "mps_runPede_rfcp_template.sh")

        # get working directory name
        currentDir = os.getcwd()
        match = re.search(re.compile('mpproduction\/mp(.+?)$', re.M|re.I),currentDir)
        if match:
            self._mps_dir_name = 'mp'+match.group(1)
        else:
            print "Current location does not seem to be a MillePede campaign directory:",
            print currentDir
            sys.exit(1)


    def _fill_general_options(self):
        """Create and fill `general_options` dictionary."""

        self._fetch_essentials()
        self._fetch_defaults()
        self._fetch_dataset_directory()


    def _create_mass_storage_directory(self):
        """
        Create MPS mass storage directory where, e.g., mille binaries are
        stored.
        """

        # set directory on eos
        self._mss_dir = self._general_options.get("massStorageDir",
                                                  "/eos/cms/store/caf/user/"
                                                  +os.environ["USER"])
        self._mss_dir = os.path.join(self._mss_dir, "MPproduction",
                                     self._mps_dir_name)

        cmd = ["mkdir", "-p", self._mss_dir]

        # create directory
        if not self._general_options.get("testMode", False):
            try:
                with open(os.devnull, "w") as dump:
                    subprocess.check_call(cmd, stdout = dump, stderr = dump)
            except subprocess.CalledProcessError:
                print "Failed to create mass storage directory:", self._mss_dir
                sys.exit(1)


    def _create_weight_configs(self):
        """Extract different weight configurations from `self._config`."""

        weight_dict = collections.OrderedDict()
        common_weights = {}

        # loop over datasets to reassign weights
        for section in self._config.sections():
            if 'general' in section:
                continue
            elif section == "weights":
                for option in self._config.options(section):
                    common_weights[option] \
                        = [x.strip() for x in
                           self._config.get(section, option).split(",")]
            elif section.startswith("dataset:"):
                name = section[8:]  # get name of dataset by stripping of "dataset:"
                if self._config.has_option(section,'weight'):
                    weight_dict[name] \
                        = [x.strip() for x in
                           self._config.get(section, "weight").split(",")]
                else:
                    weight_dict[name] = ['1.0']

        weights_list = [[(name, weight) for weight in  weight_dict[name]]
                        for name in weight_dict]

        common_weights_list = [[(name, weight)
                                for weight in  common_weights[name]]
                               for name in common_weights]
        common_weights_dicts = []
        for item in itertools.product(*common_weights_list):
            d = {}
            for name,weight in item:
                d[name] = weight
            common_weights_dicts.append(d)

        self._weight_configs = []
        for weight_conf in itertools.product(*weights_list):
            if len(common_weights) > 0:
                for common_weight in common_weights_dicts:
                    self._weight_configs.append([(dataset[0],
                                     reduce(lambda x,y: x.replace(y, common_weight[y]),
                                            common_weight, dataset[1]))
                                    for dataset in weight_conf])
            else:
                self._weight_configs.append(weight_conf)


    def _fetch_pede_settings(self):
        """Fetch 'pedesettings' from general section in `self._config`."""

        self._pede_settings \
            = ([x.strip()
                for x in self._config.get("general", "pedesettings").split(",")]
               if self._config.has_option("general", "pedesettings") else [None])


    def _create_mille_jobs(self):
        """Create the mille jobs based on the [dataset:<name>] sections."""

        first_dataset = True
        for section in self._config.sections():
            if "general" in section: continue
            elif section.startswith("dataset:"):
                self._dataset_options={}
                print "-"*60

                # set name from section-name
                self._dataset_options["name"] = section[8:]

                # extract essential variables
                for var in ("inputFileList", "collection"):
                    try:
                        self._dataset_options[var] = self._config.get(section,var)
                    except ConfigParser.NoOptionError:
                        print "No", var, "found in", section+". Please check ini-file."
                        sys.exit(1)

                # get globaltag and configTemplate. If none in section, try to get
                # default from [general] section.
                for var in ("configTemplate", "globaltag"):
                    if self._config.has_option(section,var):
                        self._dataset_options[var] = self._config.get(section,var)
                    else:
                        try:
                            self._dataset_options[var] = self._general_options[var]
                        except KeyError:
                            print "No",var,"found in ["+section+"]",
                            print "and no default in [general] section."
                            sys.exit(1)

                # extract non-essential options
                self._dataset_options["cosmicsZeroTesla"] = False
                if self._config.has_option(section,"cosmicsZeroTesla"):
                    self._dataset_options["cosmicsZeroTesla"] \
                        = self._config.getboolean(section,"cosmicsZeroTesla")

                self._dataset_options["cosmicsDecoMode"] = False
                if self._config.has_option(section,"cosmicsDecoMode"):
                    self._dataset_options["cosmicsDecoMode"] \
                        = self._config.getboolean(section,"cosmicsDecoMode")

                self._dataset_options["primaryWidth"] = -1.0
                if self._config.has_option(section,"primaryWidth"):
                    self._dataset_options["primaryWidth"] \
                        = self._config.getfloat(section,"primaryWidth")

                self._dataset_options["json"] = ""
                if self._config.has_option(section, "json"):
                    self._dataset_options["json"] = self._config.get(section,"json")
                else:
                    try:
                        self._dataset_options["json"] = self._general_options["json"]
                    except KeyError:
                        print "No json given in either [general] or ["+section+"] sections.",
                        print "Proceeding without json-file."


                # replace ${datasetdir} and other variables in inputFileList-path
                self._dataset_options["inputFileList"] \
                    = os.path.expandvars(self._dataset_options["inputFileList"])

                # replace variables in configTemplate-path, e.g. $CMSSW_BASE
                self._dataset_options["configTemplate"] \
                    = os.path.expandvars(self._dataset_options["configTemplate"])


                # Get number of jobs from lines in inputfilelist
                self._dataset_options["njobs"] = 0
                try:
                    with open(self._dataset_options["inputFileList"], "r") as filelist:
                        for line in filelist:
                            if "CastorPool" in line:
                                continue
                            # ignore empty lines
                            if not line.strip()=="":
                                self._dataset_options["njobs"] += 1
                except IOError:
                    print "Inputfilelist", self._dataset_options["inputFileList"],
                    print "does not exist."
                    sys.exit(1)
                if self._dataset_options["njobs"] == 0:
                    print "Number of jobs is 0. There may be a problem with the inputfilelist:"
                    print self._dataset_options["inputFileList"]
                    sys.exit(1)

                # Check if njobs gets overwritten in .ini-file
                if self._config.has_option(section, "njobs"):
                    if self._config.getint(section, "njobs") <= self._dataset_options["njobs"]:
                        self._dataset_options["njobs"] = self._config.getint(section, "njobs")
                    else:
                        print "'njobs' is bigger than the number of files for this",
                        print "dataset:", self._dataset_options["njobs"]
                        print "Using default."
                else:
                    print "No number of jobs specified. Using number of files in",
                    print "inputfilelist as the number of jobs."


                # Build config from template/Fill in variables
                try:
                    with open(self._dataset_options["configTemplate"],"r") as f:
                        tmpFile = f.read()
                except IOError:
                    print "The config-template called",
                    print self._dataset_options["configTemplate"], "cannot be found."
                    sys.exit(1)

                tmpFile = re.sub('setupGlobaltag\s*\=\s*[\"\'](.*?)[\"\']',
                                 'setupGlobaltag = \"'+self._dataset_options["globaltag"]+'\"',
                                 tmpFile)
                tmpFile = re.sub(re.compile("setupRunStartGeometry\s*\=\s*.*$", re.M),
                                 "setupRunStartGeometry = "+
                                 self._general_options["FirstRunForStartGeometry"], tmpFile)
                tmpFile = re.sub('setupCollection\s*\=\s*[\"\'](.*?)[\"\']',
                                 'setupCollection = \"'+self._dataset_options["collection"]+'\"',
                                 tmpFile)
                if self._dataset_options['cosmicsZeroTesla']:
                    tmpFile = re.sub(re.compile('setupCosmicsZeroTesla\s*\=\s*.*$', re.M),
                                     'setupCosmicsZeroTesla = True',
                                     tmpFile)
                if self._dataset_options['cosmicsDecoMode']:
                    tmpFile = re.sub(re.compile('setupCosmicsDecoMode\s*\=\s*.*$', re.M),
                                     'setupCosmicsDecoMode = True',
                                     tmpFile)
                if self._dataset_options['primaryWidth'] > 0.0:
                    tmpFile = re.sub(re.compile('setupPrimaryWidth\s*\=\s*.*$', re.M),
                                     'setupPrimaryWidth = '+str(self._dataset_options["primaryWidth"]),
                                     tmpFile)
                if self._dataset_options['json'] != '':
                    tmpFile = re.sub(re.compile('setupJson\s*\=\s*.*$', re.M),
                                     'setupJson = \"'+self._dataset_options["json"]+'\"',
                                     tmpFile)

                thisCfgTemplate = "tmp.py"
                with open(thisCfgTemplate, "w") as f:
                    f.write(tmpFile)


                # Set mps_setup append option for datasets following the first one
                append = "-a"
                if first_dataset:
                    append = ""
                    first_dataset = False
                    self._config_template = tmpFile
                    self._cms_process = mps_tools.get_process_object(thisCfgTemplate)
                    self._create_input_db()

                with open(thisCfgTemplate, "a") as f: f.write(self._override_gt)


                # create mps_setup command
                command = ["mps_setup.pl",
                           "-m",
                           append,
                           "-M", self._general_options["pedeMem"],
                           "-N", self._dataset_options["name"],
                           self._mille_script,
                           thisCfgTemplate,
                           self._dataset_options["inputFileList"],
                           str(self._dataset_options["njobs"]),
                           self._general_options["classInf"],
                           self._general_options["jobname"],
                           self._pede_script,
                           "cmscafuser:"+self._mss_dir]
                command = filter(lambda x: len(x.strip()) > 0, command)

                # Some output:
                print "Submitting dataset:", self._dataset_options["name"]
                print "Baseconfig:        ", self._dataset_options["configTemplate"]
                print "Collection:        ", self._dataset_options["collection"]
                if self._dataset_options["collection"] in ("ALCARECOTkAlCosmicsCTF0T",
                                                           "ALCARECOTkAlCosmicsInCollisions"):
                    print "cosmicsDecoMode:   ", self._dataset_options["cosmicsDecoMode"]
                    print "cosmicsZeroTesla:  ", self._dataset_options["cosmicsZeroTesla"]
                print "Globaltag:         ", self._dataset_options["globaltag"]
                print "Number of jobs:    ", self._dataset_options["njobs"]
                print "Inputfilelist:     ", self._dataset_options["inputFileList"]
                if self._dataset_options["json"] != "":
                    print "Jsonfile:          ", self._dataset_options["json"]
                print "Pass to mps_setup: ", " ".join(command)

                # call the command and toggle verbose output
                self._handle_process_call(command, self._args.verbose)

                # remove temporary file
                self._handle_process_call(["rm", thisCfgTemplate])

        if first_dataset:
            print "No dataset section defined in '{0}'".format(self._args.aligmentConfig)
            print "At least one section '[dataset:<name>]' is required."
            sys.exit(1)

        self._global_tag = self._dataset_options["globaltag"]
        self._first_run = self._general_options["FirstRunForStartGeometry"]


    def _create_pede_jobs(self):
        """Create pede jobs from the given input."""

        for setting in self._pede_settings:
            print
            print "="*60
            if setting is None:
                print "Creating pede job."
            else:
                print "Creating pede jobs using settings from '{0}'.".format(setting)
            for weight_conf in self._weight_configs:
                print "-"*60
                # blank weights
                self._handle_process_call(["mps_weight.pl", "-c"])

                thisCfgTemplate = "tmp.py"
                with open(thisCfgTemplate, "w") as f: f.write(self._config_template)
                if self._override_gt is None:
                    self._cms_process = mps_tools.get_process_object(thisCfgTemplate)
                    self.create_input_db()
                with open(thisCfgTemplate, "a") as f: f.write(self._override_gt)

                for name,weight in weight_conf:
                    self._handle_process_call(["mps_weight.pl", "-N", name, weight], True)

                if not self._first_pede_config:
                    # create new mergejob
                    self._handle_process_call(["mps_setupm.pl"], True)

                # read mps.db to find directory of new mergejob
                lib = mpslib.jobdatabase()
                lib.read_db()

                # short cut for jobm path
                jobm_path = os.path.join("jobData", lib.JOBDIR[-1])

                # delete old merge-config
                command = ["rm", "-f", os.path.join(jobm_path, "alignment_merge.py")]
                self._handle_process_call(command, self._args.verbose)

                # create new merge-config
                command = [
                    "mps_merge.py",
                    "-w", thisCfgTemplate,
                    os.path.join(jobm_path, "alignment_merge.py"),
                    jobm_path,
                    str(lib.nJobs),
                ]
                if setting is not None: command.extend(["-a", setting])
                print " ".join(command)
                self._handle_process_call(command, self._args.verbose)
                self._create_tracker_tree()
                if self._first_pede_config:
                    os.symlink(self._tracker_tree_path,
                               os.path.abspath(os.path.join(jobm_path,
                                                            ".TrackerTree.root")))
                    self._first_pede_config = False

                # store weights configuration
                with open(os.path.join(jobm_path, ".weights.pkl"), "wb") as f:
                    cPickle.dump(weight_conf, f, 2)

        # remove temporary file
        self._handle_process_call(["rm", thisCfgTemplate])


    def _create_additional_pede_jobs(self):
        """
        Create pede jobs in addition to already existing ones. Return GT
        override snippet.
        """

        # do some basic checks
        if not os.path.isdir("jobData"):
            print "No jobData-folder found.",
            print "Properly set up the alignment before using the -w option."
            sys.exit(1)
        if not os.path.exists("mps.db"):
            print "No mps.db found.",
            print "Properly set up the alignment before using the -w option."
            sys.exit(1)

        # check if default configTemplate is given
        try:
            config_template = self._config.get("general", "configTemplate")
        except ConfigParser.NoOptionError:
            print "No default configTemplate given in [general] section."
            print "When using -w, a default configTemplate is needed to build",
            print "a merge-config."
            sys.exit(1)

        # check if default globaltag is given
        try:
            self._global_tag = self._config.get("general", "globaltag")
        except ConfigParser.NoOptionError:
            print "No default 'globaltag' given in [general] section."
            print "When using -w, a default configTemplate is needed to build",
            print "a merge-config."
            sys.exit(1)

        try:
            self._first_run = self._config.get("general",
                                               "FirstRunForStartGeometry")
        except ConfigParser.NoOptionError:
            print "Missing mandatory option 'FirstRunForStartGeometry'",
            print "in [general] section."
            sys.exit(1)

        for section in self._config.sections():
            if section.startswith("dataset:"):
                try:
                    collection = self._config.get(section, "collection")
                    break
                except ConfigParser.NoOptionError:
                    print "Missing mandatory option 'collection' in section",
                    print "["+section+"]."
                    sys.exit(1)

        try:
            with open(config_template,"r") as f:
                tmpFile = f.read()
        except IOError:
            print "The config-template '"+config_template+"' cannot be found."
            sys.exit(1)

        tmpFile = re.sub('setupGlobaltag\s*\=\s*[\"\'](.*?)[\"\']',
                         'setupGlobaltag = \"'+self._global_tag+'\"',
                         tmpFile)
        tmpFile = re.sub('setupCollection\s*\=\s*[\"\'](.*?)[\"\']',
                         'setupCollection = \"'+collection+'\"',
                         tmpFile)
        tmpFile = re.sub(re.compile("setupRunStartGeometry\s*\=\s*.*$", re.M),
                         "setupRunStartGeometry = "+self._first_run,
                         tmpFile)
        self._config_template = tmpFile

        # first pede job exists already in this mode:
        self._first_pede_config = False
        self._create_pede_jobs()


    def _handle_process_call(self, command, verbose = False):
        """
        Wrapper around subprocess calls which treats output depending on verbosity
        level.

        Arguments:
        - `command`: list of command items
        - `verbose`: flag to turn on verbosity
        """

        call_method = subprocess.check_call if verbose else subprocess.check_output
        try:
            call_method(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print "" if verbose else e.output
            print "Failed to execute command:", " ".join(command)
            sys.exit(1)


    def _create_input_db(self):
        """
        Create sqlite file with single-IOV tags and use it to override the
        GT. If the GT is already customized by the user, the customization has
        higher priority. Creates a snippet to be appended to the configuration
        file.
        """

        run_number = int(self._first_run)
        if not run_number > 0:
            print "'FirstRunForStartGeometry' must be positive, but is", run_number
            sys.exit(1)

        input_db_name = os.path.abspath("alignment_input.db")
        tags = mps_tools.create_single_iov_db(self._check_iov_definition(),
                                              run_number, input_db_name)

        self._override_gt = ""
        for record,tag in tags.iteritems():
            if self._override_gt == "":
                self._override_gt \
                    += ("\nimport "
                        "Alignment.MillePedeAlignmentAlgorithm.alignmentsetup."
                        "SetCondition as tagwriter\n")
            self._override_gt += ("\ntagwriter.setCondition(process,\n"
                                  "       connect = \""+tag["connect"]+"\",\n"
                                  "       record = \""+record+"\",\n"
                                  "       tag = \""+tag["tag"]+"\")\n")


    def _check_iov_definition(self):
        """
        Check consistency of input alignment payloads and IOV definition.
        Returns a dictionary with the information needed to override possibly
        problematic input taken from the global tag.
        """

        print "Checking consistency of IOV definition..."
        iovs = mps_tools.make_unique_runranges(self._cms_process.AlignmentProducer)

        inputs = {
            "TrackerAlignmentRcd": None,
            "TrackerSurfaceDeformationRcd": None,
            "TrackerAlignmentErrorExtendedRcd": None,
        }

        for condition in self._cms_process.GlobalTag.toGet.value():
            if condition.record.value() in inputs:
                inputs[condition.record.value()] = {
                    "tag": condition.tag.value(),
                    "connect": ("pro"
                                if not condition.hasParameter("connect")
                                else condition.connect.value())
                }

        inputs_from_gt = [record for record in inputs if inputs[record] is None]
        inputs.update(
            mps_tools.get_tags(self._cms_process.GlobalTag.globaltag.value(),
                               inputs_from_gt))

        if int(self._first_run) != iovs[0]:     # simple consistency check
            if iovs[0] == 1 and len(iovs) == 1:
                print "Single IOV output detected in configuration and",
                print "'FirstRunForStartGeometry' is not 1."
                print "Creating single IOV output from input conditions in run",
                print self._first_run+"."
                for inp in inputs: inputs[inp]["problematic"] = True
            else:
                print "Value of 'FirstRunForStartGeometry' has to match first",
                print "defined output IOV:",
                print self._first_run, "!=", iovs[0]
                sys.exit(1)

        for inp in inputs.itervalues():
            inp["iovs"] = mps_tools.get_iovs(inp["connect"], inp["tag"])

        # check consistency of input with output
        problematic_gt_inputs = {}
        input_indices = {key: len(value["iovs"]) -1
                         for key,value in inputs.iteritems()}
        for iov in reversed(iovs):
            for inp in inputs:
                if inputs[inp].pop("problematic", False):
                    problematic_gt_inputs[inp] = inputs[inp]
                if inp in problematic_gt_inputs: continue
                if input_indices[inp] < 0:
                    print "First output IOV boundary at run", iov,
                    print "is before the first input IOV boundary at",
                    print inputs[inp]["iovs"][0], "for '"+inp+"'."
                    print "Please check your run range selection."
                    sys.exit(1)
                input_iov = inputs[inp]["iovs"][input_indices[inp]]
                if iov < input_iov:
                    if inp in inputs_from_gt:
                        problematic_gt_inputs[inp] = inputs[inp]
                        print "Found problematic input taken from global tag."
                        print "Input IOV boundary at run",input_iov,
                        print "for '"+inp+"' is within output IOV starting",
                        print "with run", str(iov)+"."
                        print "Deriving an alignment with coarse IOV",
                        print "granularity starting from finer granularity",
                        print "leads to wrong results."
                        print "A single IOV input using the IOV of",
                        print "'FirstRunForStartGeometry' ("+self._first_run+")",
                        print "is automatically created and used."
                        continue
                    print "Found input IOV boundary at run",input_iov,
                    print "for '"+inp+"' which is within output IOV starting",
                    print "with run", str(iov)+"."
                    print "Deriving an alignment with coarse IOV granularity",
                    print "starting from finer granularity leads to wrong",
                    print "results."
                    print "Please check your run range selection."
                    sys.exit(1)
                elif iov == input_iov:
                    input_indices[inp] -= 1

        # check consistency of 'TrackerAlignmentRcd' with other inputs
        input_indices = {key: len(value["iovs"]) -1
                         for key,value in inputs.iteritems()
                         if (key != "TrackerAlignmentRcd")
                         and (inp not in problematic_gt_inputs)}
        for iov in reversed(inputs["TrackerAlignmentRcd"]["iovs"]):
            for inp in input_indices:
                input_iov = inputs[inp]["iovs"][input_indices[inp]]
                if iov < input_iov:
                    print "Found input IOV boundary at run",input_iov,
                    print "for '"+inp+"' which is within 'TrackerAlignmentRcd'",
                    print "IOV starting with run", str(iov)+"."
                    print "Deriving an alignment with inconsistent IOV boundaries",
                    print "leads to wrong results."
                    print "Please check your input IOVs."
                    sys.exit(1)
                elif iov == input_iov:
                    input_indices[inp] -= 1

        print "IOV consistency check successful."
        print "-"*60

        return problematic_gt_inputs


    def _create_tracker_tree(self):
        """Method to create hidden 'TrackerTree.root'."""

        if self._global_tag is None or self._first_run is None:
            print "Trying to create the tracker tree before setting the global",
            print "tag or the run to determine the geometry IOV."
            sys.exit(1)

        config = mpsv_iniparser.ConfigData()
        config.jobDataPath = "."    # current directory
        config.globalTag = self._global_tag
        config.firstRun = self._first_run
        self._tracker_tree_path = mpsv_trackerTree.check(config)


    def _fetch_essentials(self):
        """Fetch general options from config file."""

        for var in ("classInf","pedeMem","jobname", "FirstRunForStartGeometry"):
            try:
                self._general_options[var] = self._config.get('general',var)
            except ConfigParser.NoOptionError:
                print "No", var, "found in [general] section.",
                print "Please check ini-file."
                sys.exit(1)


    def _fetch_defaults(self):
        """Fetch default general options from config file."""

        for var in ("globaltag", "configTemplate", "json", "massStorageDir",
                    "testMode"):
            try:
                self._general_options[var] = self._config.get("general", var)
            except ConfigParser.NoOptionError:
                if var == "testMode": continue
                print "No '" + var + "' given in [general] section."


    def _fetch_dataset_directory(self):
        """
        Fetch 'datasetDir' variable from general section and add it to the
        'os.environ' dictionary.
        """

        if self._config.has_option("general", "datasetdir"):
            dataset_directory = self._config.get("general", "datasetdir")
            # add it to environment for later variable expansion:
            os.environ["datasetdir"] = dataset_directory
            self._general_options["datasetdir"] = dataset_directory
        else:
            print "No datasetdir given in [general] section.",
            print "Be sure to give a full path in inputFileList."
            self._general_options["datasetdir"] = ""


################################################################################
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
