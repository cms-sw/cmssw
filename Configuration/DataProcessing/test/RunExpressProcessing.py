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
        self.dataTiers = []
        self.datasets = []
        self.alcaDataset = None
        self.globalTag = 'UNSPECIFIED::All'
        self.inputLFN = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError, msg
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError, msg
        if len(self.dataTiers) == 0:
            msg = "No --data-tiers provided, need at least one"
            raise RuntimeError, msg
        if len(self.datasets) == 0:
            msg = "No --datasets provided, need at least one"
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
        print "Writing Data Tiers :"
        for tier in self.dataTiers:
            print " => %s" % tier
        print "Into Datasets:"
        for dataset in self.datasets:
            print " => %s" % dataset
        print "Alca Dataset set to: %s" % self.alcaDataset

        try:

            process = scenario.expressProcessing(
                self.globalTag,  self.dataTiers,
                self.datasets, self.alcaDataset)
        except Exception, ex:
            msg = "Error creating Express processing config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)


        psetFile = open("RunExpressProcessingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -f FrameworkJobReport.xml RunExpressProcessingCfg.py"
        print "Now do:\n%s" % cmsRun
        



if __name__ == '__main__':
    valid = ["scenario=", "data-tiers=", "datasets=", "alca-dataset=",
             "global-tag=", "lfn="]
    usage = """RunPromptReco.py <options>"""
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
        if opt == "--data-tiers":
            expressinator.dataTiers = [
                x for x in arg.split(",") if len(x) > 0 ]
        if opt == "--datasets":
            expressinator.datasets = [
                x for x in arg.split(",") if len(x) > 0 ]
        if opt == "--alca-dataset":
            expressinator.alcaDataset = arg

    expressinator()
