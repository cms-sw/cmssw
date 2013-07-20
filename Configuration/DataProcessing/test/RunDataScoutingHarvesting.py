#!/usr/bin/env python
"""
_RunDQMHarvesting_

Test wrapper to generate a harvesting config and push it into cmsRun for
testing with a few input files etc from the command line

"""

import sys
import getopt

from Configuration.DataProcessing.GetScenario import getScenario



class RunDataScoutingHarvesting:

    def __init__(self):
        self.scenario = "DataScouting"
        self.dataset = None
        self.run = None
        self.globalTag = 'UNSPECIFIED::All'
        self.inputLFN = None

    def __call__(self):
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
            process = scenario.dqmHarvesting(self.dataset, self.run,
                                                      self.globalTag)
            
        except Exception, ex:
            msg = "Error creating Harvesting config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)


        psetFile = open("RunDataScoutingHarvestingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -j FrameworkJobReport.xml RunDataScoutingHarvestingCfg.py"
        print "Now do:\n%s" % cmsRun
        



if __name__ == '__main__':
    valid = ["scenario=", "run=", "dataset=",
             "global-tag=", "lfn="]
    usage = """RunDataScoutingHarvesting.py <options>"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    dataScoutingHarvester = RunDataScoutingHarvesting()

    for opt, arg in opts:
        if opt == "--global-tag":
            dataScoutingHarvester.globalTag = arg
        if opt == "--lfn" :
            dataScoutingHarvester.inputLFN = arg
        if opt == "--run":
            dataScoutingHarvester.run = arg
        if opt == "--dataset":
            dataScoutingHarvester.dataset = arg

    dataScoutingHarvester()
