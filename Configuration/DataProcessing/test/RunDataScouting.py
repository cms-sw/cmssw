#!/usr/bin/env python
"""
_RunDataScouting_

Test wrapper to generate a data scouting config

17-7-2012
I silently comply to the history of the code in the package.

"""

import sys
import getopt

from Configuration.DataProcessing.GetScenario import getScenario



class RunDataScouting:

    def __init__(self):
       self.scenario = "DataScouting" #what else...
       self.globalTag = None
       self.inputLFN = None

    def __call__(self):
        if self.globalTag == None:
            msg = "No --globaltag specified"
            raise RuntimeError, msg
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError, msg

        #try:
        scenario = getScenario(self.scenario)
        #except Exception, ex:
        #    msg = "Error getting Scenario implementation for %s\n" % (
        #        self.scenario,)
        #    msg += str(ex)
        #    raise RuntimeError, msg

        print "Retrieved Scenario: %s" % self.scenario
        print "Using Global Tag: %s" % self.globalTag

        dataTiers = ["DQM"]

        # get config with specified output
        process = scenario.promptReco(globalTag = self.globalTag, writeTiers = dataTiers)

        #except NotImplementedError, ex:
            #print "This scenario does not support DataScouting:\n"
            #return
        #except Exception, ex:
            #msg = "Error creating Prompt Reco config:\n"
            #msg += str(ex)
            #raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        psetFile = open("RunDataScoutingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunDataScoutingCfg.py"
        print "Now do:\n%s" % cmsRun



if __name__ == '__main__':
    valid = ["scenario=", "global-tag=", "lfn="]
    usage = \
"""
RunDataScouting.py <options>

Where options are:
 --global-tag=GlobalTag
 --lfn=/store/input/lfn


Example:
python RunDataScouting.py --global-tag GLOBALTAG::ALL --lfn=/store/whatever

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    dataScoutator = RunDataScouting()

    for opt, arg in opts:
        if opt == "--global-tag":
            dataScoutator.globalTag = arg
        if opt == "--lfn" :
            dataScoutator.inputLFN = arg
        

    dataScoutator()
