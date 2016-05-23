#!/usr/bin/env python

#############################################################################
##  Builds the config-templates from the universal config-template for each
##  dataset specified in .ini-file that is passed to this script as argument.
##  Then calls mps_setup.pl for all datasets.
##
##  Usage:
##     mps_alisetup.py [-h] [-v] myconfig.ini
##

import argparse
import os
import re
import subprocess
import ConfigParser
import sys
import itertools
import collections
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
from Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper import checked_out_MPS


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



# ------------------------------------------------------------------------------
# set up argument parser and config parser

helpEpilog ='''Builds the config-templates from a universal config-template for each
dataset specified in .ini-file that is passed to this script.
Then calls mps_setup.pl for all datasets.'''
parser = argparse.ArgumentParser(
        description='Setup the alignment as configured in the alignment_config file.',
        epilog=helpEpilog)
# optional argmuent: verbose (toggles output of mps_setup)
parser.add_argument('-v', '--verbose', action='store_true',
                    help='display detailed output of mps_setup')
# optional argmuent: weight (new weights for additional merge job)
parser.add_argument('-w', '--weight', action='store_true',
                    help='create an additional mergejob with (possibly new) weights from .ini-config')
# positional argument: config file
parser.add_argument('alignmentConfig', action='store',
                    help='name of the .ini config file that specifies the datasets to be used')
# parse argument
args = parser.parse_args()
aligmentConfig = args.alignmentConfig

# parse config file
config = ConfigParser.ConfigParser()
config.optionxform = str    # default would give lowercase options -> not wanted
config.read(aligmentConfig)



#------------------------------------------------------------------------------
# construct directories

# set variables that are not too specific (millescript, pedescript, etc.)
mpsScriptsDir = os.path.join("src", "Alignment", "MillePedeAlignmentAlgorithm", "scripts")
if checked_out_MPS()[0]:
    mpsScriptsDir = os.path.join(os.environ["CMSSW_BASE"], mpsScriptsDir)
else:
    mpsScriptsDir = os.path.join(os.environ["CMSSW_RELEASE_BASE"], mpsScriptsDir)
milleScript = os.path.join(mpsScriptsDir, "mps_runMille_template.sh")
pedeScript  = os.path.join(mpsScriptsDir, "mps_runPede_rfcp_template.sh")

# get working directory name
currentDir = os.getcwd()
mpsdirname = ''
match = re.search(re.compile('mpproduction\/mp(.+?)$', re.M|re.I),currentDir)
if match:
    mpsdirname = 'mp'+match.group(1)
else:
    print 'there seems to be a problem to determine the current directory name:',currentDir
    sys.exit(1)

# set directory on eos
mssDir = '/store/caf/user/'+os.environ['USER']+'/MPproduction/'+mpsdirname

# create directory on eos if it doesn't exist already
eos = '/afs/cern.ch/project/eos/installation/cms/bin/eos.select'
os.system(eos+' mkdir -p '+mssDir)



#------------------------------------------------------------------------------
# read general-section
generalOptions = {}

# essential variables
for var in ['classInf','pedeMem','jobname']:
    try:
        generalOptions[var] = config.get('general',var)
    except ConfigParser.NoOptionError:
        print "No", var, "found in [general] section. Please check ini-file."
        raise SystemExit

# check if datasetdir is given
generalOptions['datasetdir'] = ''
if config.has_option('general','datasetdir'):
    generalOptions['datasetdir'] = config.get('general','datasetdir')
    # add it to environment for later variable expansion:
    os.environ["datasetdir"] = generalOptions['datasetdir']
else:
    print "No datasetdir given in [general] section.",
    print "Be sure to give a full path in inputFileList."

# check for default options
for var in ['globaltag','configTemplate','json']:
    try:
        generalOptions[var] = config.get('general',var)
    except ConfigParser.NoOptionError:
        print 'No default', var, 'given in [general] section.'



pedesettings = ([x.strip() for x in config.get("general", "pedesettings").split(",")]
                if config.has_option("general", "pedesettings") else [None])

weight_confs = get_weight_configs(config)


#------------------------------------------------------------------------------
# -w option: Get new weights from .ini and create additional mergejob with these
if args.weight:

    # do some basic checks
    if not os.path.isdir("jobData"):
        print "No jobData-folder found. Properly set up the alignment before using the -w option."
        raise SystemExit
    if not os.path.exists("mps.db"):
        print "No mps.db found. Properly set up the alignment before using the -w option."
        raise SystemExit

    # check if default configTemplate is given
    try:
        configTemplate = config.get('general','configTemplate')
    except ConfigParser.NoOptionError:
        print 'No default configTemplate given in [general] section.'
        print 'When using -w, a default configTemplate is needed to build a merge-config.'
        raise SystemExit

    # check if default globaltag is given
    try:
        globalTag = config.get('general','globaltag')
    except ConfigParser.NoOptionError:
        print "No default 'globaltag' given in [general] section."
        print "When using -w, a default configTemplate is needed to build a merge-config."
        raise SystemExit

    try:
        with open(configTemplate,"r") as f:
            tmpFile = f.read()
    except IOError:
        print "The config-template '"+configTemplate+"' cannot be found."
        raise SystemExit

    tmpFile = re.sub('setupGlobaltag\s*\=\s*[\"\'](.*?)[\"\']',
                     'setupGlobaltag = \"'+globalTag+'\"',
                     tmpFile)

    thisCfgTemplate = "tmp.py"
    with open(thisCfgTemplate, "w") as f:
        f.write(tmpFile)

    for setting in pedesettings:
        print
        print "="*60
        if setting is None:
            print "Creating pede job."
        else:
            print "Creating pede jobs using settings from '{0}'.".format(setting)
        for weight_conf in weight_confs:
            print "-"*60
            # blank weights
            os.system("mps_weight.pl -c > /dev/null")

            for name,weight in weight_conf:
                os.system("mps_weight.pl -N "+name+" "+weight)

            # create new mergejob
            os.system("mps_setupm.pl")

            # read mps.db to find directory of new mergejob
            lib = mpslib.jobdatabase()
            lib.read_db()

            # delete old merge-config
            command = "rm -f jobData/"+lib.JOBDIR[-1]+"/alignment_merge.py"
            print command
            os.system(command)

            # create new merge-config
            command = ("mps_merge.py -w "+thisCfgTemplate+" jobData/"+
                       lib.JOBDIR[-1]+"/alignment_merge.py jobData/"+
                       lib.JOBDIR[-1]+" "+str(lib.nJobs))
            if setting is not None: command += " -a "+setting
            print command
            if args.verbose:
                subprocess.call(command, stderr=subprocess.STDOUT, shell=True)
            else:
                with open(os.devnull, 'w') as FNULL:
                    subprocess.call(command, stdout=FNULL,
                                    stderr=subprocess.STDOUT, shell=True)

    # remove temporary file
    os.system("rm "+thisCfgTemplate)
    sys.exit()


#------------------------------------------------------------------------------
# loop over dataset-sections
firstDataset = True
for section in config.sections():
    if 'general' in section:
        continue
    elif section.startswith("dataset:"):
        datasetOptions={}
        print "-"*60

        # set name from section-name
        datasetOptions['name'] = section[8:]

        # extract essential variables
        for var in ['inputFileList','collection']:
            try:
                datasetOptions[var] = config.get(section,var)
            except ConfigParser.NoOptionError:
                print 'No', var, 'found in', section+'. Please check ini-file.'
                raise SystemExit

        # get globaltag and configTemplate. If none in section, try to get default from [general] section.
        for var in ['configTemplate','globaltag']:
            if config.has_option(section,var):
                datasetOptions[var] = config.get(section,var)
            else:
                try:
                    datasetOptions[var] = generalOptions[var]
                except KeyError:
                    print "No",var,"found in ["+section+"]",
                    print "and no default in [general] section."
                    raise SystemExit

        # extract non-essential options
        datasetOptions['cosmicsZeroTesla'] = False
        if config.has_option(section,'cosmicsZeroTesla'):
            datasetOptions['cosmicsZeroTesla'] = config.getboolean(section,'cosmicsZeroTesla')

        datasetOptions['cosmicsDecoMode'] = False
        if config.has_option(section,'cosmicsDecoMode'):
            datasetOptions['cosmicsDecoMode'] = config.getboolean(section,'cosmicsDecoMode')

        datasetOptions['primaryWidth'] = -1.0
        if config.has_option(section,'primaryWidth'):
            datasetOptions['primaryWidth'] = config.getfloat(section,'primaryWidth')

        datasetOptions['json'] = ''
        if config.has_option(section, 'json'):
            datasetOptions['json'] = config.get(section,'json')
        else:
            try:
                datasetOptions['json'] = generalOptions['json']
            except KeyError:
                print "No json given in either [general] or ["+section+"] sections.",
                print "Proceeding without json-file."


        # replace '${datasetdir}' and other variables in inputFileList-path
        datasetOptions['inputFileList'] = os.path.expandvars(datasetOptions['inputFileList'])

        # replace variables in configTemplate-path, e.g. $CMSSW_BASE
        datasetOptions['configTemplate'] = os.path.expandvars(datasetOptions['configTemplate'])


        # Get number of jobs from lines in inputfilelist
        datasetOptions['njobs'] = 0
        try:
            with open(datasetOptions['inputFileList'],'r') as filelist:
                for line in filelist:
                    if 'CastorPool' in line:
                        continue
                    # ignore empty lines
                    if not line.strip()=='':
                        datasetOptions['njobs'] += 1
        except IOError:
            print 'Inputfilelist', datasetOptions['inputFileList'], 'does not exist.'
            raise SystemExit
        if datasetOptions['njobs'] == 0:
            print 'Number of jobs is 0. There may be a problem with the inputfilelist:'
            print datasetOptions['inputFileList']
            raise SystemExit

        # Check if njobs gets overwritten in .ini-file
        if config.has_option(section,'njobs'):
            if config.getint(section,'njobs')<=datasetOptions['njobs']:
                datasetOptions['njobs'] = config.getint(section,'njobs')
            else:
                print 'njobs is bigger than the default',datasetOptions['njobs'],'. Using default.'
        else:
            print 'No number of jobs specified. Using number of files in inputfilelist as the number of jobs.'


        # Build config from template/Fill in variables
        try:
            with open(datasetOptions['configTemplate'],'r') as INFILE:
                tmpFile = INFILE.read()
        except IOError:
            print 'The config-template called',datasetOptions['configTemplate'],'cannot be found.'
            raise SystemExit

        tmpFile = re.sub('setupGlobaltag\s*\=\s*[\"\'](.*?)[\"\']',
                         'setupGlobaltag = \"'+datasetOptions['globaltag']+'\"',
                         tmpFile)
        tmpFile = re.sub('setupCollection\s*\=\s*[\"\'](.*?)[\"\']',
                         'setupCollection = \"'+datasetOptions['collection']+'\"',
                         tmpFile)
        if datasetOptions['cosmicsZeroTesla']:
            tmpFile = re.sub(re.compile('setupCosmicsZeroTesla\s*\=\s*.*$', re.M),
                             'setupCosmicsZeroTesla = True',
                             tmpFile)
        if datasetOptions['cosmicsDecoMode']:
            tmpFile = re.sub(re.compile('setupCosmicsDecoMode\s*\=\s*.*$', re.M),
                             'setupCosmicsDecoMode = True',
                             tmpFile)
        if datasetOptions['primaryWidth'] > 0.0:
            tmpFile = re.sub(re.compile('setupPrimaryWidth\s*\=\s*.*$', re.M),
                             'setupPrimaryWidth = '+str(datasetOptions['primaryWidth']),
                             tmpFile)
        if datasetOptions['json'] != '':
            tmpFile = re.sub(re.compile('setupJson\s*\=\s*.*$', re.M),
                             'setupJson = \"'+datasetOptions['json']+'\"',
                             tmpFile)

        thisCfgTemplate = 'tmp.py'
        with open(thisCfgTemplate, 'w') as OUTFILE:
            OUTFILE.write(tmpFile)


        # Set mps_setup append option for datasets following the first one
        append = ' -a'
        if firstDataset:
            append = ''
            firstDataset = False
            configTemplate = tmpFile

        # create mps_setup command
        command = 'mps_setup.pl -m%s -M %s -N %s %s %s %s %d %s %s %s cmscafuser:%s' % (
              append,
              generalOptions['pedeMem'],
              datasetOptions['name'],
              milleScript,
              thisCfgTemplate,
              datasetOptions['inputFileList'],
              datasetOptions['njobs'],
              generalOptions['classInf'],
              generalOptions['jobname'],
              pedeScript,
              mssDir)
        # Some output:
        print 'Submitting dataset:', datasetOptions['name']
        print 'Baseconfig:        ', datasetOptions['configTemplate']
        print 'Collection:        ', datasetOptions['collection']
        if datasetOptions['collection']=='ALCARECOTkAlCosmicsCTF0T':
            print 'cosmicsDecoMode:   ', datasetOptions['cosmicsDecoMode']
            print 'cosmicsZeroTesla:  ', datasetOptions['cosmicsZeroTesla']
        print 'Globaltag:         ', datasetOptions['globaltag']
        print 'Number of jobs:    ', datasetOptions['njobs']
        print 'Inputfilelist:     ', datasetOptions['inputFileList']
        if datasetOptions['json'] != '':
            print 'Jsonfile:      ', datasetOptions['json']
        print 'Pass to mps_setup: ', command

        # call the command and toggle verbose output
        if args.verbose:
            subprocess.call(command, stderr=subprocess.STDOUT, shell=True)
        else:
            with open(os.devnull, 'w') as FNULL:
                subprocess.call(command, stdout=FNULL,
                                stderr=subprocess.STDOUT, shell=True)

        # remove temporary file
        os.system("rm "+thisCfgTemplate)

if firstDataset:
    print "No dataset section defined in '{0}'".format(aligmentConfig)
    print "At least one section '[dataset:<name>]' is required."
    sys.exit(1)

firstPedeConfig = True
for setting in pedesettings:
    print
    print "="*60
    if setting is None:
        print "Creating pede job."
    else:
        print "Creating pede jobs using settings from '{0}'.".format(setting)
    for weight_conf in weight_confs:
        print "-"*60
        # blank weights
        os.system("mps_weight.pl -c > /dev/null")

        for name,weight in weight_conf:
            os.system("mps_weight.pl -N "+name+" "+weight)

        if firstPedeConfig:
            firstPedeConfig = False
        else:
            # create new mergejob
            os.system("mps_setupm.pl")

        # read mps.db to find directory of new mergejob
        lib = mpslib.jobdatabase()
        lib.read_db()

        # delete old merge-config
        command = "rm -f jobData/"+lib.JOBDIR[-1]+"/alignment_merge.py"
        print command
        os.system(command)

        thisCfgTemplate = "tmp.py"
        with open(thisCfgTemplate, "w") as f:
            f.write(configTemplate)

        # create new merge-config
        command = ("mps_merge.py -w "+thisCfgTemplate+" jobData/"+lib.JOBDIR[-1]+
                   "/alignment_merge.py jobData/"+lib.JOBDIR[-1]+" "+
                   str(lib.nJobs))
        if setting is not None: command += " -a "+setting
        print command
        if args.verbose:
            subprocess.call(command, stderr=subprocess.STDOUT, shell=True)
        else:
            with open(os.devnull, 'w') as FNULL:
                subprocess.call(command, stdout=FNULL,
                                stderr=subprocess.STDOUT, shell=True)

    # remove temporary file
    os.system("rm "+thisCfgTemplate)
