#!/usr/bin/env python
"""
_RunExpressProcessing_

Test wrapper to generate an express processing config and actually push
it into cmsRun for testing with a few input files etc from the command line

"""

import sys
import getopt

from Configuration.DataProcessing.GetScenario import getScenario



class RunExpressProcessing:

    def __init__(self):
        self.scenario = None
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

        try:
            process = scenario.expressProcessing(self.globalTag, writeTiers = ['RECO'])
        except NotImplementedError, ex:
            print "This scenario does not support Express Processing:\n"
            return
        except Exception, ex:
            msg = "Error creating Express Processing config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)

        psetFile = open("RunExpressProcessingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunExpressProcessingCfg.py"
        print "Now do:\n%s" % cmsRun



if __name__ == '__main__':
    valid = ["scenario=", "global-tag=", "lfn="]
    usage = \
"""
RunExpressProcessing.py <options>

Where options are:
 --scenario=ScenarioName
 --global-tag=GlobalTag
 --lfn=/store/input/lfn


Example:
python2.4 RunPromptReco.py --scenario=Cosmics --global-tag GLOBALTAG::ALL --lfn=/store/whatever

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    expressinator = RunExpressProcessing()

    for opt, arg in opts:
        if opt == "--scenario":
            expressinator.scenario = arg
        if opt == "--global-tag":
            expressinator.globalTag = arg
        if opt == "--lfn" :
            expressinator.inputLFN = arg

    expressinator()
