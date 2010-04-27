#!/usr/bin/env python
"""
_RunPromptReco_

Test wrapper to generate a reco config and actually push it into cmsRun for
testing with a few input files etc from the command line

"""

import sys
import getopt

from Configuration.DataProcessing.GetScenario import getScenario



class RunPromptReco:

    def __init__(self):
        self.scenario = None
        self.writeReco = False
        self.writeAlca = False
        self.writeAlcareco = False
        self.writeAod = False
        self.globalTag = None
        self.inputLFN = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError, msg
        if self.globalTag == None:
            msg = "No --globaltag specified"
            raise RuntimeError, msg
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError, msg

        try:
            scenario = getScenario(self.scenario)
        except Exception, ex:
            msg = "Error getting Scenario implementation for %s\n" % (
                self.scenario,)
            msg += str(ex)
            raise RuntimeError, msg

        print "Retrieved Scenario: %s" % self.scenario
        print "Using Global Tag: %s" % self.globalTag

        dataTiers = []
        if self.writeReco:
            dataTiers.append("RECO")
            print "Configuring to Write out Reco..."
        if self.writeAlcareco:
            dataTiers.append("ALCARECO")
            print "Configuring to Write out Alca..."
        if self.writeAod:
            dataTiers.append("AOD")
            print "Configuring to Write out Aod..."

        try:
            if len(dataTiers) > 0:
                # use command line options for data-tiers
                process = scenario.promptReco(self.globalTag,
                                              dataTiers)
            else:
                # use default data-tiers for current scenario
                process = scenario.promptReco(self.globalTag)
        except NotImplementedError, ex:
            print "This scenario does not support Prompt Reco:\n"
            return
        except Exception, ex:
            msg = "Error creating Prompt Reco config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        psetFile = open("RunPromptRecoCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunPromptRecoCfg.py"
        print "Now do:\n%s" % cmsRun



if __name__ == '__main__':
    valid = ["scenario=", "reco", "alcareco", "aod",
             "global-tag=", "lfn="]
    usage = \
"""
RunPromptReco.py <options>

Where options are:
 --scenario=ScenarioName
 --reco (to enable RECO output)
 --alcareco (to enable ALCARECO output)
 --aod (to enable AOD output)
 --global-tag=GlobalTag
 --lfn=/store/input/lfn


Example:
python2.4 RunPromptReco.py --scenario=Cosmics --reco --aod --global-tag GLOBALTAG::ALL --lfn=/store/whatever

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    recoinator = RunPromptReco()

    for opt, arg in opts:
        if opt == "--scenario":
            recoinator.scenario = arg
        if opt == "--reco":
            recoinator.writeReco = True
        if opt == "--alcareco":
            recoinator.writeAlcareco = True
        if opt == "--aod":
            recoinator.writeAod = True
        if opt == "--global-tag":
            recoinator.globalTag = arg
        if opt == "--lfn" :
            recoinator.inputLFN = arg
        

    recoinator()
