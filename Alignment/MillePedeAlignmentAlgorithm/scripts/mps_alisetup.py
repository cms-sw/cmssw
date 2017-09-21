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


def main(argv = None):
    """Main routine. Not called, if this module is loaded via `import`.

    Arguments:
    - `argv`: Command line arguments passed to the script.
    """

    if argv == None:
        argv = sys.argv[1:]

    # --------------------------------------------------------------------------
    # set up argument parser and config parser

    helpEpilog ="""Builds the config-templates from a universal config-template
    for each dataset specified in .ini-file that is passed to this script.  Then
    calls mps_setup.pl for all datasets."""
    parser = argparse.ArgumentParser(
        description="Setup the alignment as configured in the alignment_config file.",
        epilog=helpEpilog)
    # optional argmuent: verbose (toggles output of mps_setup)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="display detailed output of mps_setup")
    # optional argmuent: weight (new weights for additional merge job)
    parser.add_argument("-w", "--weight", action="store_true",
                        help=("create an additional mergejob with (possibly new)"
                              " weights from .ini-config"))
    # positional argument: config file
    parser.add_argument("alignmentConfig",
                        help=("name of the .ini config file that specifies the "
                              "datasets to be used"))
    # parse argument
    args = parser.parse_args(argv)
    aligmentConfig = args.alignmentConfig

    # parse config file
    config = ConfigParser.ConfigParser()
    config.optionxform = str # default would give lowercase options -> not wanted
    config.read(aligmentConfig)



    #---------------------------------------------------------------------------
    # construct directories

    # set variables that are not too specific (millescript, pedescript, etc.)
    mpsTemplates = os.path.join("src", "Alignment",
                                "MillePedeAlignmentAlgorithm", "templates")
    if checked_out_MPS()[0]:
        mpsTemplates = os.path.join(os.environ["CMSSW_BASE"], mpsTemplates)
    else:
        mpsTemplates = os.path.join(os.environ["CMSSW_RELEASE_BASE"], mpsTemplates)
    mille_script = os.path.join(mpsTemplates, "mps_runMille_template.sh")
    pede_script  = os.path.join(mpsTemplates, "mps_runPede_rfcp_template.sh")

    # get working directory name
    currentDir = os.getcwd()
    mpsdirname = ''
    match = re.search(re.compile('mpproduction\/mp(.+?)$', re.M|re.I),currentDir)
    if match:
        mpsdirname = 'mp'+match.group(1)
    else:
        print "Current location does not seem to be a MillePede campaign directory:",
        print currentDir
        sys.exit(1)


    # read general-section
    general_options = {}
    general_options.update(fetch_essentials(config))
    general_options.update(fetch_defaults(config))
    general_options['datasetdir'] = fetch_dataset_directory(config)

    mss_dir = create_mass_storage_directory(mpsdirname, general_options)
    pede_settings = fetch_pede_settings(config)
    weight_confs = get_weight_configs(config)


    if args.weight:
        global_tag, first_run, override_gt \
            = create_additional_pede_jobs(config, pede_settings, weight_confs, args)
    else:
        # loop over dataset-sections
        dataset_options, config_template, override_gt \
            = create_mille_jobs(config, general_options,
                                mille_script, pede_script,
                                mss_dir, args)
        global_tag = dataset_options["globaltag"]
        first_run = general_options["FirstRunForStartGeometry"]

        create_pede_jobs(config_template = config_template,
                         pede_settings = pede_settings,
                         weight_confs = weight_confs,
                         global_tag = global_tag,
                         first_run = first_run,
                         args = args,
                         override_gt = override_gt,
                         first_pede_config = True)

    if override_gt.strip() != "":
        print "="*60
        msg = ("Overriding global tag with single-IOV tags extracted from '{}' "
               "for run number '{}'.".format(global_tag, first_run))
        print msg
        print "-"*60
        print override_gt


def handle_process_call(command, verbose = False):
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


def get_weight_configs(config):
    """Extracts different weight configurations from `config`.

    Arguments:
    - `config`: ConfigParser object containing the alignment configuration
    """

    weight_dict = collections.OrderedDict()
    common_weights = {}

    # loop over datasets to reassign weights
    for section in config.sections():
        if 'general' in section:
            continue
        elif section == "weights":
            for option in config.options(section):
                common_weights[option] = [x.strip() for x in
                                          config.get(section, option).split(",")]
        elif section.startswith("dataset:"):
            name = section[8:]  # get name of dataset by stripping of "dataset:"
            if config.has_option(section,'weight'):
                weight_dict[name] = [x.strip() for x in
                                     config.get(section, "weight").split(",")]
            else:
                weight_dict[name] = ['1.0']

    weights_list = [[(name, weight) for weight in  weight_dict[name]]
                    for name in weight_dict]

    common_weights_list = [[(name, weight) for weight in  common_weights[name]]
                           for name in common_weights]
    common_weights_dicts = []
    for item in itertools.product(*common_weights_list):
        d = {}
        for name,weight in item:
            d[name] = weight
        common_weights_dicts.append(d)

    configs = []
    for weight_conf in itertools.product(*weights_list):
        if len(common_weights) > 0:
            for common_weight in common_weights_dicts:
                configs.append([(dataset[0],
                                 reduce(lambda x,y: x.replace(y, common_weight[y]),
                                        common_weight, dataset[1]))
                                for dataset in weight_conf])
        else:
            configs.append(weight_conf)

    return configs


def create_input_db(cms_process, run_number):
    """
    Create sqlite file with single-IOV tags and use it to override the GT. If
    the GT is already customized by the user, the customization has higher
    priority. Returns a snippet to be appended to the configuration file

    Arguments:
    - `cms_process`: cms.Process object
    - `run_number`: run from which to extract the alignment payloads
    """

    run_number = int(run_number)
    if not run_number > 0:
        print "'FirstRunForStartGeometry' must be positive, but is", run_number
        sys.exit(1)

    input_db_name = os.path.abspath("alignment_input.db")
    tags = mps_tools.create_single_iov_db(
        check_iov_definition(cms_process, run_number),
        run_number, input_db_name)

    result = ""
    for record,tag in tags.iteritems():
        if result == "":
            result += ("\nimport "
                       "Alignment.MillePedeAlignmentAlgorithm.alignmentsetup."
                       "SetCondition as tagwriter\n")
        result += ("\ntagwriter.setCondition(process,\n"
                   "       connect = \""+tag["connect"]+"\",\n"
                   "       record = \""+record+"\",\n"
                   "       tag = \""+tag["tag"]+"\")\n")

    return result


def check_iov_definition(cms_process, first_run):
    """
    Check consistency of input alignment payloads and IOV definition.
    Returns a dictionary with the information needed to override possibly
    problematic input taken from the global tag.

    Arguments:
    - `cms_process`: cms.Process object containing the CMSSW configuration
    - `first_run`: first run for start geometry
    """

    print "Checking consistency of IOV definition..."
    iovs = mps_tools.make_unique_runranges(cms_process.AlignmentProducer)

    inputs = {
        "TrackerAlignmentRcd": None,
        "TrackerSurfaceDeformationRcd": None,
        "TrackerAlignmentErrorExtendedRcd": None,
    }

    for condition in cms_process.GlobalTag.toGet.value():
        if condition.record.value() in inputs:
            inputs[condition.record.value()] = {
                "tag": condition.tag.value(),
                "connect": ("pro"
                            if not condition.hasParameter("connect")
                            else condition.connect.value())
            }

    inputs_from_gt = [record for record in inputs if inputs[record] is None]
    inputs.update(mps_tools.get_tags(cms_process.GlobalTag.globaltag.value(),
                                     inputs_from_gt))


    if first_run != iovs[0]:     # simple consistency check
        if iovs[0] == 1 and len(iovs) == 1:
            print "Single IOV output detected in configuration and",
            print "'FirstRunForStartGeometry' is not 1."
            print "Creating single IOV output from input conditions in run",
            print str(first_run)+"."
            for inp in inputs: inputs[inp]["problematic"] = True
        else:
            print "Value of 'FirstRunForStartGeometry' has to match first",
            print "defined output IOV:",
            print first_run, "!=", iovs[0]
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
                    print "for '"+inp+"' is within output IOV starting with",
                    print "run", str(iov)+"."
                    print "Deriving an alignment with coarse IOV granularity",
                    print "starting from finer granularity leads to wrong",
                    print "results."
                    print "A single IOV input using the IOV of",
                    print "'FirstRunForStartGeometry' ("+str(first_run)+") is",
                    print "automatically created and used."
                    continue
                print "Found input IOV boundary at run",input_iov,
                print "for '"+inp+"' which is within output IOV starting with",
                print "run", str(iov)+"."
                print "Deriving an alignment with coarse IOV granularity",
                print "starting from finer granularity leads to wrong results."
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


def create_mass_storage_directory(mps_dir_name, general_options):
    """Create MPS mass storage directory where, e.g., mille binaries are stored.

    Arguments:
    - `mps_dir_name`: campaign name
    - `general_options`: general options dictionary
    """

    # set directory on eos
    mss_dir = general_options.get("massStorageDir",
                                  "/eos/cms/store/caf/user/"+os.environ["USER"])
    mss_dir = os.path.join(mss_dir, "MPproduction", mps_dir_name)

    cmd = ["mkdir", "-p", mss_dir]

    # create directory
    if not general_options.get("testMode", False):
        try:
            with open(os.devnull, "w") as dump:
                subprocess.check_call(cmd, stdout = dump, stderr = dump)
        except subprocess.CalledProcessError:
            print "Failed to create mass storage directory:", mss_dir
            sys.exit(1)

    return mss_dir


def create_tracker_tree(global_tag, first_run):
    """Method to create hidden 'TrackerTree.root'.

    Arguments:
    - `global_tag`: global tag from which the tracker geometry is taken
    - `first_run`: run to specify IOV within `global_tag`
    """

    config = mpsv_iniparser.ConfigData()
    config.jobDataPath = "."    # current directory
    config.globalTag = global_tag
    config.firstRun = first_run
    return mpsv_trackerTree.check(config)


def fetch_essentials(config):
    """Fetch general options from `config` file.

    Arguments:
    - `config`: ConfigParser object
    """

    essentials = {}
    for var in ("classInf","pedeMem","jobname", "FirstRunForStartGeometry"):
        try:
            essentials[var] = config.get('general',var)
        except ConfigParser.NoOptionError:
            print "No", var, "found in [general] section. Please check ini-file."
            sys.exit(1)

    return essentials


def fetch_defaults(config):
    """Fetch default general options from `config` file.

    Arguments:
    - `config`: ConfigParser object
    """

    defaults = {}
    for var in ("globaltag", "configTemplate", "json", "massStorageDir",
                "testMode"):
        try:
            defaults[var] = config.get('general',var)
        except ConfigParser.NoOptionError:
            if var == "testMode": continue
            print "No '" + var + "' given in [general] section."

    return defaults


def fetch_dataset_directory(config):
    """Fetch 'datasetDir' variable from general section and add it to the
       'os.environ' dictionary.

    Arguments:
    - `config`: ConfigParser object
    """

    if config.has_option('general','datasetdir'):
        dataset_directory = config.get('general','datasetdir')
        # add it to environment for later variable expansion:
        os.environ["datasetdir"] = dataset_directory
        return dataset_directory
    else:
        print "No datasetdir given in [general] section.",
        print "Be sure to give a full path in inputFileList."
        return ""


def fetch_pede_settings(config):
    """Fetch 'pedesettings' from general section in `config` file.

    Arguments:
    - `config`: ConfigParser object
    """

    return ([x.strip() for x in config.get("general", "pedesettings").split(",")]
            if config.has_option("general", "pedesettings") else [None])


def create_mille_jobs(config, general_options, mille_script, pede_script,
                      mss_dir, args):
    """Create the mille jobs based on the [dataset:<name>] sections.

    Arguments:
    - `config`: ConfigParser object
    - `general_options`: dictionary containing general options
    - `mille_script`: template to create mille execution script
    - `pede_script`: template to create pede execution script
    - `mss_dir`: path to mass storage directory
    - `args`: command line arguments
    """

    first_dataset = True
    override_gt = ""
    for section in config.sections():
        if "general" in section: continue
        elif section.startswith("dataset:"):
            dataset_options={}
            print "-"*60

            # set name from section-name
            dataset_options["name"] = section[8:]

            # extract essential variables
            for var in ("inputFileList", "collection"):
                try:
                    dataset_options[var] = config.get(section,var)
                except ConfigParser.NoOptionError:
                    print "No", var, "found in", section+". Please check ini-file."
                    sys.exit(1)

            # get globaltag and configTemplate. If none in section, try to get
            # default from [general] section.
            for var in ("configTemplate", "globaltag"):
                if config.has_option(section,var):
                    dataset_options[var] = config.get(section,var)
                else:
                    try:
                        dataset_options[var] = general_options[var]
                    except KeyError:
                        print "No",var,"found in ["+section+"]",
                        print "and no default in [general] section."
                        sys.exit(1)

            # extract non-essential options
            dataset_options["cosmicsZeroTesla"] = False
            if config.has_option(section,"cosmicsZeroTesla"):
                dataset_options["cosmicsZeroTesla"] = config.getboolean(section,"cosmicsZeroTesla")

            dataset_options["cosmicsDecoMode"] = False
            if config.has_option(section,"cosmicsDecoMode"):
                dataset_options["cosmicsDecoMode"] = config.getboolean(section,"cosmicsDecoMode")

            dataset_options["primaryWidth"] = -1.0
            if config.has_option(section,"primaryWidth"):
                dataset_options["primaryWidth"] = config.getfloat(section,"primaryWidth")

            dataset_options["json"] = ""
            if config.has_option(section, "json"):
                dataset_options["json"] = config.get(section,"json")
            else:
                try:
                    dataset_options["json"] = general_options["json"]
                except KeyError:
                    print "No json given in either [general] or ["+section+"] sections.",
                    print "Proceeding without json-file."


            # replace ${datasetdir} and other variables in inputFileList-path
            dataset_options["inputFileList"] = os.path.expandvars(dataset_options["inputFileList"])

            # replace variables in configTemplate-path, e.g. $CMSSW_BASE
            dataset_options["configTemplate"] = os.path.expandvars(dataset_options["configTemplate"])


            # Get number of jobs from lines in inputfilelist
            dataset_options["njobs"] = 0
            try:
                with open(dataset_options["inputFileList"], "r") as filelist:
                    for line in filelist:
                        if "CastorPool" in line:
                            continue
                        # ignore empty lines
                        if not line.strip()=="":
                            dataset_options["njobs"] += 1
            except IOError:
                print "Inputfilelist", dataset_options["inputFileList"], "does not exist."
                sys.exit(1)
            if dataset_options["njobs"] == 0:
                print "Number of jobs is 0. There may be a problem with the inputfilelist:"
                print dataset_options["inputFileList"]
                sys.exit(1)

            # Check if njobs gets overwritten in .ini-file
            if config.has_option(section, "njobs"):
                if config.getint(section, "njobs") <= dataset_options["njobs"]:
                    dataset_options["njobs"] = config.getint(section, "njobs")
                else:
                    print "'njobs' is bigger than the number of files for this",
                    print "dataset:", dataset_options["njobs"]
                    print "Using default."
            else:
                print "No number of jobs specified. Using number of files in",
                print "inputfilelist as the number of jobs."


            # Build config from template/Fill in variables
            try:
                with open(dataset_options["configTemplate"],"r") as INFILE:
                    tmpFile = INFILE.read()
            except IOError:
                print "The config-template called",
                print dataset_options["configTemplate"], "cannot be found."
                sys.exit(1)

            tmpFile = re.sub('setupGlobaltag\s*\=\s*[\"\'](.*?)[\"\']',
                             'setupGlobaltag = \"'+dataset_options["globaltag"]+'\"',
                             tmpFile)
            tmpFile = re.sub(re.compile("setupRunStartGeometry\s*\=\s*.*$", re.M),
                             "setupRunStartGeometry = "+
                             general_options["FirstRunForStartGeometry"], tmpFile)
            tmpFile = re.sub('setupCollection\s*\=\s*[\"\'](.*?)[\"\']',
                             'setupCollection = \"'+dataset_options["collection"]+'\"',
                             tmpFile)
            if dataset_options['cosmicsZeroTesla']:
                tmpFile = re.sub(re.compile('setupCosmicsZeroTesla\s*\=\s*.*$', re.M),
                                 'setupCosmicsZeroTesla = True',
                                 tmpFile)
            if dataset_options['cosmicsDecoMode']:
                tmpFile = re.sub(re.compile('setupCosmicsDecoMode\s*\=\s*.*$', re.M),
                                 'setupCosmicsDecoMode = True',
                                 tmpFile)
            if dataset_options['primaryWidth'] > 0.0:
                tmpFile = re.sub(re.compile('setupPrimaryWidth\s*\=\s*.*$', re.M),
                                 'setupPrimaryWidth = '+str(dataset_options["primaryWidth"]),
                                 tmpFile)
            if dataset_options['json'] != '':
                tmpFile = re.sub(re.compile('setupJson\s*\=\s*.*$', re.M),
                                 'setupJson = \"'+dataset_options["json"]+'\"',
                                 tmpFile)

            thisCfgTemplate = "tmp.py"
            with open(thisCfgTemplate, "w") as OUTFILE:
                OUTFILE.write(tmpFile)


            # Set mps_setup append option for datasets following the first one
            append = "-a"
            if first_dataset:
                append = ""
                first_dataset = False
                config_template = tmpFile
                cms_process = mps_tools.get_process_object(thisCfgTemplate)
                override_gt = create_input_db(cms_process,
                                             general_options["FirstRunForStartGeometry"])

            with open(thisCfgTemplate, "a") as f: f.write(override_gt)


            # create mps_setup command
            command = ["mps_setup.pl",
                       "-m",
                       append,
                       "-M", general_options["pedeMem"],
                       "-N", dataset_options["name"],
                       mille_script,
                       thisCfgTemplate,
                       dataset_options["inputFileList"],
                       str(dataset_options["njobs"]),
                       general_options["classInf"],
                       general_options["jobname"],
                       pede_script,
                       "cmscafuser:"+mss_dir]
            command = filter(lambda x: len(x.strip()) > 0, command)

            # Some output:
            print "Submitting dataset:", dataset_options["name"]
            print "Baseconfig:        ", dataset_options["configTemplate"]
            print "Collection:        ", dataset_options["collection"]
            if dataset_options["collection"] in ("ALCARECOTkAlCosmicsCTF0T",
                                                "ALCARECOTkAlCosmicsInCollisions"):
                print "cosmicsDecoMode:   ", dataset_options["cosmicsDecoMode"]
                print "cosmicsZeroTesla:  ", dataset_options["cosmicsZeroTesla"]
            print "Globaltag:         ", dataset_options["globaltag"]
            print "Number of jobs:    ", dataset_options["njobs"]
            print "Inputfilelist:     ", dataset_options["inputFileList"]
            if dataset_options["json"] != "":
                print "Jsonfile:          ", dataset_options["json"]
            print "Pass to mps_setup: ", " ".join(command)

            # call the command and toggle verbose output
            handle_process_call(command, args.verbose)

            # remove temporary file
            handle_process_call(["rm", thisCfgTemplate])

    if first_dataset:
        print "No dataset section defined in '{0}'".format(aligmentConfig)
        print "At least one section '[dataset:<name>]' is required."
        sys.exit(1)

    return dataset_options, config_template, override_gt



def create_pede_jobs(config_template, pede_settings, weight_confs, global_tag,
                     first_run, args,
                     override_gt = None, first_pede_config = False):
    """Create pede jobs from the given input. Return GT override snippet.

    Arguments:
    - `config_template`: configuration config template
    - `pede_settings`: list of names of pede setting files
    - `weight_confs`: dictionary containing the weights to be applied to the
                      different input datasets
    - `global_tag`: global tag to be used for the pede job
    - `first_run`: first run for the start geometry
    - `args`: command line arguments
    - `override_gt`: snippet to appended in case of conditions to be overridden
    - `first_pede_config`: flag indicating if the first pede config still has to
                           be created
    """

    for setting in pede_settings:
        print
        print "="*60
        if setting is None:
            print "Creating pede job."
        else:
            print "Creating pede jobs using settings from '{0}'.".format(setting)
        for weight_conf in weight_confs:
            print "-"*60
            # blank weights
            handle_process_call(["mps_weight.pl", "-c"])

            thisCfgTemplate = "tmp.py"
            with open(thisCfgTemplate, "w") as f: f.write(config_template)
            if override_gt is None:
                cms_process = mps_tools.get_process_object(thisCfgTemplate)
                override_gt = create_input_db(cms_process, first_run)
            with open(thisCfgTemplate, "a") as f: f.write(override_gt)

            for name,weight in weight_conf:
                handle_process_call(["mps_weight.pl", "-N", name, weight], True)

            if not first_pede_config:
                # create new mergejob
                handle_process_call(["mps_setupm.pl"], True)

            # read mps.db to find directory of new mergejob
            lib = mpslib.jobdatabase()
            lib.read_db()

            # short cut for jobm path
            jobm_path = os.path.join("jobData", lib.JOBDIR[-1])

            # delete old merge-config
            command = ["rm", "-f", os.path.join(jobm_path, "alignment_merge.py")]
            handle_process_call(command, args.verbose)

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
            handle_process_call(command, args.verbose)
            tracker_tree_path = create_tracker_tree(global_tag, first_run)
            if first_pede_config:
                os.symlink(tracker_tree_path,
                           os.path.abspath(os.path.join(jobm_path,
                                                        ".TrackerTree.root")))
                first_pede_config = False

            # store weights configuration
            with open(os.path.join(jobm_path, ".weights.pkl"), "wb") as f:
                cPickle.dump(weight_conf, f, 2)

    # remove temporary file
    handle_process_call(["rm", thisCfgTemplate])

    return override_gt


def create_additional_pede_jobs(config, pede_settings, weight_confs, args):
    """
    Create pede jobs in addition to already existing ones. Return GT override
    snippet.

    Arguments:
    - `config`: ConfigParser object
    - `pede_settings`: list of names of pede setting files
    - `weight_confs`: dictionary containing the weights to be applied to the
                      different input datasets
    - `args`: command line arguments
    """


    # do some basic checks
    if not os.path.isdir("jobData"):
        print "No jobData-folder found. Properly set up the alignment before using the -w option."
        sys.exit(1)
    if not os.path.exists("mps.db"):
        print "No mps.db found. Properly set up the alignment before using the -w option."
        sys.exit(1)

    # check if default configTemplate is given
    try:
        config_template = config.get('general','configTemplate')
    except ConfigParser.NoOptionError:
        print 'No default configTemplate given in [general] section.'
        print 'When using -w, a default configTemplate is needed to build a merge-config.'
        sys.exit(1)

    # check if default globaltag is given
    try:
        global_tag = config.get('general','globaltag')
    except ConfigParser.NoOptionError:
        print "No default 'globaltag' given in [general] section."
        print "When using -w, a default configTemplate is needed to build a merge-config."
        sys.exit(1)

    try:
        first_run = config.get("general", "FirstRunForStartGeometry")
    except ConfigParser.NoOptionError:
        print "Missing mandatory option 'FirstRunForStartGeometry' in [general] section."
        sys.exit(1)

    for section in config.sections():
        if section.startswith("dataset:"):
            try:
                collection = config.get(section, "collection")
                break
            except ConfigParser.NoOptionError:
                print "Missing mandatory option 'collection' in section ["+section+"]."
                sys.exit(1)

    try:
        with open(config_template,"r") as f:
            tmpFile = f.read()
    except IOError:
        print "The config-template '"+config_template+"' cannot be found."
        sys.exit(1)

    tmpFile = re.sub('setupGlobaltag\s*\=\s*[\"\'](.*?)[\"\']',
                     'setupGlobaltag = \"'+global_tag+'\"',
                     tmpFile)
    tmpFile = re.sub('setupCollection\s*\=\s*[\"\'](.*?)[\"\']',
                     'setupCollection = \"'+collection+'\"',
                     tmpFile)
    tmpFile = re.sub(re.compile("setupRunStartGeometry\s*\=\s*.*$", re.M),
                     "setupRunStartGeometry = "+first_run,
                     tmpFile)

    return global_tag, first_run, create_pede_jobs(config_template = tmpFile,
                                                   pede_settings = pede_settings,
                                                   weight_confs = weight_confs,
                                                   global_tag = global_tag,
                                                   first_run = first_run,
                                                   args = args,
                                                   override_gt = None,
                                                   first_pede_config = False)


################################################################################
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
