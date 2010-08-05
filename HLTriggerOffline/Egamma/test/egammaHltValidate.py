#!/usr/bin/env python
# adapted to python and extended based on a shell script by Michael Anderson
#
# To use, type for example:
#  hltValidate 3_5_5
#
# This script runs validation on new relval samples.
# Requires version of CMSSW given.
#
# Michael Anderson
# Sept 15, 2009

import sys, os, shutil, getopt
#----------------------------------------------------------------------
# parameters
#----------------------------------------------------------------------
# CMSSW Module to check out & compile
module="HLTriggerOffline/Egamma"

# Config file in module to run with cmsRun
configFile="test/test_cfg.py"

# Root file name outputted by running module
outputRootFile="DQM_V0001_HLT_R000000001.root"


# Datasets to run on
knownDatasets = {
    "photonJet" : {
        "dataset": "/RelValPhotonJets_Pt_10/CMSSW_%(version)s*/GEN-SIM-DIGI-RAW-HLTDEBUG",
        "output":  "GammaJet_%(version)s.root", 
        },
    
    "zee" : {
        "dataset": "/RelValZEE/CMSSW_%(version)s*/GEN-SIM-DIGI-RAW-HLTDEBUG",
        "output":  "ZEE_%(version)s.root",
        },

    "wen" : {
        "dataset": "/RelValWE/CMSSW_%(version)s*/GEN-SIM-DIGI-RAW-HLTDEBUG",
        "output":  "WEN_%(version)s.root",
        },
} 

#----------------------------------------------------------------------

def execCmd(cmd):
    retval = os.system(cmd)
    if retval != 0:
        raise Exception("failed to execute command '" + cmd + "', exit status = " + str(retval))

#----------------------------------------------------------------------

# code based on PhysicsTools.PatAlgos.tools.helpers.MassSearchReplaceAnyInputTagVisitor
# to replace the process names of all input tags found in sequences.
#
# necessary e.g. when one has to run the HLT and thus must later on
# use a different process name

class ReplaceProcessNameOfInputTags(object):
    """Visitor that travels within a cms.Sequence and replaces 
       It will climb down within PSets, VPSets and VInputTags to find its target.

       Useful e.g. for replacing the process names of all input tags where the 
       process name was specified explicitly.
    """

    #----------------------------------------
    def __init__(self,origProcessName,newProcessName,verbose=False):
        self._origProcessName = origProcessName
        self._newProcessName  = newProcessName
        # self._moduleName   = ''
        self._verbose=verbose

    #----------------------------------------

    def doIt(self,pset,base):
        if isinstance(pset, cms._Parameterizable):
            for name in pset.parameters_().keys():
                # if I use pset.parameters_().items() I get copies of the parameter values
                # so I can't modify the nested pset
                value = getattr(pset,name) 
                type = value.pythonTypeName()
                if type == 'cms.PSet':  
                    self.doIt(value,base+"."+name)
                elif type == 'cms.VPSet':
                    for (i,ps) in enumerate(value): self.doIt(ps, "%s.%s[%d]"%(base,name,i) )
                elif type == 'cms.VInputTag':
                    for (i,n) in enumerate(value): 
                         # VInputTag can be declared as a list of strings, so ensure that n is formatted correctly
                         n = self.standardizeInputTagFmt(n)
                         if self._verbose:print "FOUND TAG:",value[i]

                         if value[i].processName == self._origProcessName:
                             if self._verbose: print "REPLACING"
                             value[i].processName = self._newProcessName
                         else:
                             if self._verbose: print "NOT REPLACING"

                elif type == 'cms.InputTag':
                    if self._verbose:print "FOUND TAG:",value                        

                    if value.processName == self._origProcessName:
                        if self._verbose:print "REPLACING"
                        value.processName = self._newProcessName
                    else:
                        if self._verbose:print "NOT REPLACING"

    #----------------------------------------
    @staticmethod 
    def standardizeInputTagFmt(inputTag):
       ''' helper function to ensure that the InputTag is defined as cms.InputTag(str) and not as a plain str '''
       if not isinstance(inputTag, cms.InputTag):
          return cms.InputTag(inputTag)
       return inputTag

    #----------------------------------------
    def enter(self,visitee):
        label = ''
        try:    label = visitee.label()
        except AttributeError: label = '<Module not in a Process>'
        self.doIt(visitee, label)

    #----------------------------------------
    def leave(self,visitee):
        pass

    #----------------------------------------

#----------------------------------------------------------------------

def findCMSSWreleaseDir(version):
    """ runs scramv1 list to find the directory of the given CMSSW release.

    Sometimes it happens that there is more than one line in the scram output
    for the same release (and even the same directory). In general, just
    the first matching line is returned.
    
    """

    import re

    if not version.startswith("CMSSW_"):
        version = "CMSSW_" + version

    for line in os.popen('scramv1 list -c CMSSW').readlines():

        line = line.split('\n')[0].strip()

        project, release, directory = re.split('\s+',line)

        if release == version:
            return directory

#----------------------------------------------------------------------
def findDataSetFromSampleName(sampleSpec, version):
    """ from the given sample specification (e.g. photonJet), tries to get
    the relval dataset from DBS for the given CMSSW version.

    If more than one sample is found, the user is prompted
    to select one.
    """

    # Find the dataset in DBS using command. This actually
    # could find more than one dataset.

    datasetToSearchFor= knownDatasets[sampleSpec]['dataset'] % { "version": version }

    dbs_cmd = 'dbs search --query="find dataset where dataset=' + datasetToSearchFor + ' and site=srm-cms.cern.ch" | grep "HLTDEBUG"'

    cmssw_release_dir = findCMSSWreleaseDir(version)

    cmd_parts = [
        'cd ' + cmssw_release_dir,
        "eval `scramv1 runtime -sh`",
        "cd - > /dev/null",   # this seems to print a line in some cases
        dbs_cmd,
        ]

    allDatasetsToCheck=os.popen("  && ".join(cmd_parts)).readlines()
    allDatasetsToCheck = [ x.strip() for x in allDatasetsToCheck ]

    if len(allDatasetsToCheck) == 1:
        datasetToCheck = allDatasetsToCheck[0]
    elif len(allDatasetsToCheck) == 0:
        print "failed to find dataset in dbs"
        print
        print "dbs command was:"
        print dbs_cmd
        sys.exit(1)
    else:
        # more than one dataset found
        print "found the following matching datasets, please select one:"

        for i in range(len(allDatasetsToCheck)):
            print "  %2d: %s" % (i, allDatasetsToCheck[i])

        print "your choice:",
        choice = sys.stdin.readline()
        choice = int(choice)

        datasetToCheck = allDatasetsToCheck[choice]

        print "selected",datasetToCheck

    ###################################


    ###################################
    # Make sure dataset was found
    print "Looked for dataset matching " + datasetToSearchFor

    print "found"
    print "  ",datasetToCheck
    print

    return datasetToCheck

#----------------------------------------------------------------------
def createProjectArea(version):
    """creates a new scram project area for the given release
    and chdirs to it """

    print "Setting up CMSSW_" + version + " environment"
    execCmd("scramv1 project CMSSW CMSSW_" + version)
    os.chdir("CMSSW_" + version + "/src")


#----------------------------------------------------------------------

def ensureProjectAreaNotExisting(version):
    # refuse to run if the release area exists already
    # (can mix tags and samples etc.)

    project_dir = "CMSSW_" + version

    if os.path.exists(project_dir):
        print>>sys.stderr, "the project directory " + project_dir + " already exists."
        print>>sys.stderr,"Refusing to continue as this might cause unexpected results."
        sys.exit(1)

#----------------------------------------------------------------------
def usage():

    print >> sys.stderr,"""

    usage: """ + os.path.basename(sys.argv[0]) + """ [ options ]  sample cmssw-version

         e.g. """ + os.path.basename(sys.argv[0]) + """ photonJet 3_5_6



    """

    sys.exit(1)

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
from optparse import OptionParser

parser = OptionParser("""

  usage: %prog [options] sample cmssw-version

    e.g. %prog photonJet 3_5_6

  Produces the histogram files for E/gamma path validation.
"""
)

parser.add_option("--file",
                  dest="direct_input_files",
                  default = [],
                  type="str",
                  action="append", # append to list
                  help="run directly from the ROOT file given. Option can be specified multiple times.",
                  metavar="FILE")


parser.add_option("--hlt-process",
                  dest="hlt_process_name",
                  default = None,
                  type="str",
                  help="Specify the name of the HLT process. Useful e.g. when running on a file produced by yourself with a different process name.",
                  metavar="PROC")

parser.add_option("--cvstag",
                  dest="cvstag",
                  default = "HEAD",
                  type="str",
                  help="CVS tag to be used for module " + module + ". Default is to use the HEAD revision.",
                  metavar="TAG")


parser.add_option("--cfg-add",
                  dest="cfg_add",
                  default = [],
                  type="str",
                  action="append", # append to list
                  help="line to add to the generated cmsRun configuration file. Can be specified several times",
                  metavar="CFG_LINE")


(options, ARGV) = parser.parse_args()

sampleSpec = None

#----------------------------------------

if len(options.direct_input_files) == 0:
    if len(ARGV) != 2:
        print >> sys.stderr, \
            "Provide sample and CMSSW version!" +\
            "  example: " +\
            os.path.basename(sys.argv[0]) + " photonJet 3_5_6"

        print >> sys.stderr
        print >> sys.stderr,"known samples are: " + " ".join(knownDatasets.keys())
        print >> sys.stderr
        sys.exit(1)

    sampleSpec = ARGV[0]
    version=ARGV[1]

    ensureProjectAreaNotExisting(version)

    # check whether we know the specified sample
    if not knownDatasets.has_key(sampleSpec):
        print >> sys.stderr,"unknown sample " + sampleSpec + ", known samples are: " + " ".join(knownDatasets.keys())
        sys.exit(1)

    createProjectArea(version)

    datasetToCheck = findDataSetFromSampleName(sampleSpec, version)    

    # Get the file names in the dataset path, and format it for python files
    print "\n\nGetting file names for"
    print "  ",datasetToCheck

    cmssw_release_dir = findCMSSWreleaseDir(version)
    cmd_parts = [
        'cd ' + cmssw_release_dir,
        "eval `scramv1 runtime -sh`",
        "cd -",
        "dbs lsf --path=" + datasetToCheck,
        ]

    FILES=os.popen(" && ".join(cmd_parts)).readlines()
    FILES=[ x.strip() for x in FILES ]
    FILES=[ x for x in FILES if x.endswith('.root') ]

else:
    # a single input file was specified
    if len(ARGV) != 1:
        print >> sys.stderr,"if a input files were specified directly, you must provide exactly one non-option argument (the CMSSW version)"
        sys.exit(1)

    version=ARGV[0]

    FILES = options.direct_input_files[:]

    ensureProjectAreaNotExisting(version)

    datasetToCheck = "(undefined dataset)"

    createProjectArea(version)

###################################
# Check out module and build it

print "Checking out tag '" + options.cvstag + "' of " + module

execCmd("cvs -Q co -r " + options.cvstag + " " + module)
execCmd("scramv1 b")
os.chdir(module)

# Place file names in python config file
print "Placing into " + configFile + " and copying to test_cfg_new.py"

#----------------------------------------
# append things to the config file
#----------------------------------------
fout = open(configFile,"a")
print >> fout,"process.source.fileNames = " + str(FILES)
print >> fout,"process.post.dataSet = cms.untracked.string('" + datasetToCheck +"')"


# replace all HLT process names by something
# else if specified by the user

if options.hlt_process_name != None:
    # ugly code ahead, may disturb some viewers...
    #
    # dump the source code of the replacing code into
    # the CMSSW python configuration file
    import inspect

    print >> fout,"#----------------------------------------"
    print >> fout,"# replace explicit specifications of HLT process name by " + options.hlt_process_name
    print >> fout,"#----------------------------------------"
    print >> fout, inspect.getsource(ReplaceProcessNameOfInputTags)
    print >> fout
    print >> fout, "for seq in process.sequences.values():"
    print >> fout, """    seq.visit(ReplaceProcessNameOfInputTags("HLT","%s"))""" % options.hlt_process_name

# check for additional configuration text specified

if len(options.cfg_add) > 0:
    print >> fout
    print >> fout,"#----------------------------------------"

    for line in options.cfg_add:
        print >> fout,"# additional string specified on the command line"
        print >> fout,line
        
    print >> fout
    print >> fout,"#----------------------------------------"

fout.close()

#----------------------------------------
shutil.copy(configFile, "test_cfg_new.py")
logfile = os.path.join(os.getcwd(),"log")
print "Starting cmsRun test_cfg_new.py >& " + logfile
execCmd("eval `scramv1 runtime -sh` && cmsRun test_cfg_new.py >& " + logfile)

# check whether the expected output file was created
# and rename it 

if os.path.exists(outputRootFile):

    if sampleSpec != None:
        # a sample (e.g. wen or zee etc.) was specified

        renameOutputTo=knownDatasets[sampleSpec]['output'] % { "version" : version }

        shutil.move(outputRootFile, renameOutputTo)
        print "Created"
        print "  ",os.getcwd() + "/" + renameOutputTo
    else:
        print "Created"
        print "  ",os.getcwd() + "/" + outputRootFile

else: 

    print "cmsRun failed to create " + outputRootFile
    print "See log file:"
    print "   ",os.getcwd() + "/log"
    
