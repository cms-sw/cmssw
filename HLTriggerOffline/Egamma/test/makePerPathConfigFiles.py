#!/usr/bin/env python

#----------------------------------------------------------------------
#
# takes the MC menu and generates python configuration
# files for the EmDQM module (one per supported path)
#
# see an example file at HLTriggerOffline/Egamma/python/HLT_Ele17_SW_TighterEleIdIsol_L1RDQM_cfi.py
#----------------------------------------------------------------------

from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import HLTriggerOffline.Egamma.EgammaHLTValidationUtils as EgammaHLTValidationUtils
import sys, os
import six

# prefix for printouts
# msgPrefix = "[" + os.path.basename(__file__) + "]"
msgPrefix = ''



#----------------------------------------------------------------------

def makeOnePath(path, isFastSim):
    """ given a path object, returns the python text to be written
    to a _cff.py file"""

    # name of the HLT path
    pathName = path.label_()

    # we currently exclude a few 'problematic' paths (for which we
    # don't have a full recipe how to produce a monitoring path
    # for them).
    #
    # we exclude paths which contain EDFilters which we don't know
    # how to handle in the DQM modules
    moduleCXXtypes = EgammaHLTValidationUtils.getCXXTypesOfPath(refProcess,path)
    # print >> sys.stderr,"module types:", moduleCXXtypes

    hasProblematicType = False

    for problematicType in [
        # this list was collected empirically
        'HLTEgammaTriggerFilterObjectWrapper', 
        'EgammaHLTPhotonTrackIsolationProducersRegional',
        ]:

        if problematicType in moduleCXXtypes:
            print(msgPrefix +  "SKIPPING PATH",pathName,"BECAUSE DON'T KNOW HOW TO HANDLE A MODULE WITH C++ TYPE",problematicType, file=sys.stderr)
            return None

    # print >> sys.stderr,msgPrefix, "adding E/gamma HLT dqm module for path",pathName

    dqmModuleName = pathName
    if isFastSim:
        dqmModuleName = dqmModuleName + "FastSim"

    dqmModuleName = dqmModuleName + "_DQM"

    # global dqmModule

    dqmModule = EgammaHLTValidationUtils.EgammaDQMModuleMaker(refProcess, pathName,
                                                              thisCategoryData['genPid'],        # type of generated particle
                                                              thisCategoryData['numGenerated']   # number of generated particles
                                                              ).getResult()


    return dqmModuleName + " = " + repr(dqmModule)
    

    

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
# parse command line options
from optparse import OptionParser

parser = OptionParser(
    """
   usage: %prog [options] output_dir
   creates a set of files configuring the EmDQM module
   which can then be included from HLTriggerOffline/Egamma/python/EgammaValidation_cff.py

   """
)
(options, ARGV) = parser.parse_args()

isFastSim = False

#----------------------------------------

if len(ARGV) != 1:
    print("must specify exactly one non-option argument: the output directory", file=sys.stderr)
    sys.exit(1)

outputDir = ARGV.pop(0)

if os.path.exists(outputDir):
    print("output directory " + outputDir + " already exists, refusing to overwrite it / files in it", file=sys.stderr)
    sys.exit(1)
    


#----------------------------------------
# compose the DQM anlyser paths
#----------------------------------------


# maps from Egamma HLT path category to number of type and number of generated
# particles required for the histogramming
configData = {
    "singleElectron": { "genPid" : 11, "numGenerated" : 1,},
    "doubleElectron": { "genPid" : 11, "numGenerated" : 2 },
    "singlePhoton":   { "genPid" : 22, "numGenerated" : 1 },
    "doublePhoton":   { "genPid" : 22, "numGenerated" : 2 },
    }


egammaValidators = []
egammaValidatorsFS = []

pathToPythonText = {}


#--------------------
# a 'reference' process to take (and analyze) the HLT menu from
#--------------------
refProcess = cms.Process("REF")

if isFastSim:
    refProcess.load("FastSimulation.Configuration.HLT_GRun_cff")
else:
    refProcess.load("HLTrigger.Configuration.HLT_GRun_cff")

#--------------------

pathsByCategory = EgammaHLTValidationUtils.findEgammaPaths(refProcess)

os.mkdir(outputDir)
allPathsWritten = []



for hltPathCategory, thisCategoryData in six.iteritems(configData):

    # get the HLT path objects for this category
    paths = pathsByCategory[hltPathCategory]

    # fix: if there are no paths for some reason,
    # provide some dummy objects which we can delete
    # after the loop over the paths 
    path = None
    dqmModule = None

    for path in paths:

        pathName = path.label_()

        res = makeOnePath(path, isFastSim)

        if res == None:
            continue

        res = res.splitlines()

        res = [
            "#----------------------------------------",
            "# path " + pathName,
            "#----------------------------------------",
            "",
            "import FWCore.ParameterSet.Config as cms",
            "",
            ] + res

        outputFname = os.path.join(outputDir,pathName + "_DQM_cfi.py")
        assert(not os.path.exists(outputFname))
        fout = open(outputFname,"w")
        for line in res:
            print(line, file=fout)
        fout.close()

        print("wrote",outputFname, file=sys.stderr)
        allPathsWritten.append(pathName)

    # end of loop over paths

# end of loop over analysis types (single electron, ...)

print("generated the following paths:", file=sys.stderr)
for pathName in sorted(allPathsWritten):
    print("  " + pathName)
