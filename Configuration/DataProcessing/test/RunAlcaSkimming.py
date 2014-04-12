#!/usr/bin/env python
"""
_RunAlcaSkimming_

Test wrapper to generate an alca skimming config and push it into cmsRun for
testing with a few input files etc from the command line

"""

import sys
import getopt

from Configuration.DataProcessing.GetScenario import getScenario



class RunAlcaSkimming:

    def __init__(self):
        self.scenario = None
        self.skims = []
        self.inputLFN = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError, msg
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError, msg

        if len(self.skims) == 0:
            msg = "No --skims provided, need at least one"
            raise RuntimeError, msg

        if self.globalTag == None:
            msg = "No --global-tag specified"
            raise RuntimeError, msg

        try:
            scenario = getScenario(self.scenario)
        except Exception, ex:
            msg = "Error getting Scenario implementation for %s\n" % (
                self.scenario,)
            msg += str(ex)
            raise RuntimeError, msg

        print "Retrieved Scenario: %s" % self.scenario
        print "Creating ALCA skimming config with skims:"
        for skim in self.skims:
            print " => %s" % skim
            
        try:
            process = scenario.alcaSkim(self.skims, globaltag = self.globalTag)
        except NotImplementedError, ex:
            print "This scenario does not support Alca Skimming:\n"
            return
        except Exception, ex:
            msg = "Error creating Alca Skimming config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        psetFile = open("RunAlcaSkimmingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunAlcaSkimmingCfg.py"
        print "Now do:\n%s" % cmsRun



if __name__ == '__main__':
    valid = ["scenario=", "skims=", "lfn=","global-tag="]

    usage = \
"""
RunAlcaSkimming.py <options>

Where options are:
 --scenario=ScenarioName
 --lfn=/store/input/lfn
 --skims=comma,separated,list
 --global-tag=GlobalTag

Example:
python2.4 RunAlcaSkimming.py --scenario=Cosmics --lfn=/store/whatever --skims=MuAlStandAloneCosmics

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    skimmer = RunAlcaSkimming()

    for opt, arg in opts:
        if opt == "--scenario":
            skimmer.scenario = arg
        if opt == "--lfn" :
            skimmer.inputLFN = arg
        if opt == "--skims":
            skimmer.skims = [ x for x in arg.split(',') if len(x) > 0 ]
        if opt == "--global-tag":
            skimmer.globalTag = arg

    skimmer()
