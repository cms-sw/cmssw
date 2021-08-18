#!/usr/bin/env python3
"""
_RunAlcaSkimming_

Test wrapper to generate an alca skimming config and push it into cmsRun for
testing with a few input files etc from the command line

"""
from __future__ import print_function

import sys
import getopt
import pickle

from Configuration.DataProcessing.GetScenario import getScenario



class RunAlcaSkimming:

    def __init__(self):
        self.scenario = None
        self.skims = []
        self.inputLFN = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError(msg)
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError(msg)

        if len(self.skims) == 0:
            msg = "No --skims provided, need at least one"
            raise RuntimeError(msg)

        if self.globalTag == None:
            msg = "No --global-tag specified"
            raise RuntimeError(msg)

        try:
            scenario = getScenario(self.scenario)
        except Exception as ex:
            msg = "Error getting Scenario implementation for %s\n" % (
                self.scenario,)
            msg += str(ex)
            raise RuntimeError(msg)

        print("Retrieved Scenario: %s" % self.scenario)
        print("Creating ALCA skimming config with skims:")
        for skim in self.skims:
            print(" => %s" % skim)
            
        try:
            process = scenario.alcaSkim(self.skims, globaltag = self.globalTag)
        except NotImplementedError as ex:
            print("This scenario does not support Alca Skimming:\n")
            return
        except Exception as ex:
            msg = "Error creating Alca Skimming config:\n"
            msg += str(ex)
            raise RuntimeError(msg)

        process.source.fileNames.append(self.inputLFN)

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        pklFile = open("RunAlcaSkimmingCfg.pkl", "wb")
        psetFile = open("RunAlcaSkimmingCfg.py", "w")
        try:
            pickle.dump(process, pklFile, protocol=0)
            psetFile.write("import FWCore.ParameterSet.Config as cms\n")
            psetFile.write("import pickle\n")
            psetFile.write("handle = open('RunAlcaSkimmingCfg.pkl','rb')\n")
            psetFile.write("process = pickle.load(handle)\n")
            psetFile.write("handle.close()\n")
            psetFile.close()
        except Exception as ex:
            print("Error writing out PSet:")
            print(traceback.format_exc())
            raise ex
        finally:
            psetFile.close()
            pklFile.close()

        cmsRun = "cmsRun -e RunAlcaSkimmingCfg.py"
        print("Now do:\n%s" % cmsRun)



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
    except getopt.GetoptError as ex:
        print(usage)
        print(str(ex))
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
