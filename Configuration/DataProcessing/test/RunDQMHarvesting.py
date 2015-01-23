#!/usr/bin/env python
"""
_RunDQMHarvesting_

Test wrapper to generate a harvesting config and push it into cmsRun for
testing with a few input files etc from the command line

"""

import sys
import getopt

from Configuration.DataProcessing.GetScenario import getScenario



class RunDQMHarvesting:

    def __init__(self):
        self.scenario = None
        self.dataset = None
        self.run = None
        self.globalTag = 'UNSPECIFIED::All'
        self.inputLFN = None
        self.dqmio = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError, msg
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError, msg
        
        if self.run == None:
            msg = "No --run specified"
            raise RuntimeError, msg
        
        if self.dataset == None:
            msg = "No --dataset specified"
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
        print "Dataset: %s" % self.dataset
        print "Run: %s" % self.run
        
        
        try:
            kwds = {}
            if not self.dqmio is None:
                kwds['newDQMIO'] = self.dqmio

            process = scenario.dqmHarvesting(self.dataset, self.run,
                                             self.globalTag, **kwds)
            
        except Exception, ex:
            msg = "Error creating Harvesting config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)


        psetFile = open("RunDQMHarvestingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -j FrameworkJobReport.xml RunDQMHarvestingCfg.py"
        print "Now do:\n%s" % cmsRun
        



if __name__ == '__main__':
    valid = ["scenario=", "run=", "dataset=",
             "global-tag=", "lfn=", "dqmio"]
    usage = """RunDQMHarvesting.py <options>"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    harvester = RunDQMHarvesting()

    for opt, arg in opts:
        if opt == "--scenario":
            harvester.scenario = arg
        if opt == "--global-tag":
            harvester.globalTag = arg
        if opt == "--lfn" :
            harvester.inputLFN = arg
        if opt == "--run":
            harvester.run = arg
        if opt == "--dataset":
            harvester.dataset = arg
        if opt == "--dqmio":
            harvester.dqmio = True

    harvester()
