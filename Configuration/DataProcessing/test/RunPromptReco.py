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
        self.writeAod = False
        self.globalTag = 'UNSPECIFIED::All'
        self.inputLFN = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
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
        if self.writeAlca:
            dataTiers.append("ALCA")
            print "Configuring to Write out Alca..."
        if self.writeAod:
            dataTiers.append("AOD")
            print "Configuring to Write out Aod..."

        try:
            process = scenario.promptReco(self.globalTag,
                                          dataTiers)
        except Exception, ex:
            msg = "Error creating Prompt Reco config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)


        psetFile = open("RunPromptRecoCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -f FrameworkJobReport.xml RunPromptRecoCfg.py"
        print "Now do:\n%s" % cmsRun
        



if __name__ == '__main__':
    valid = ["scenario=", "reco", "alca", "aod",
             "global-tag=", "lfn="]
    usage = """RunPromptReco.py <options>"""
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
        if opt == "--alca":
            recoinator.writeAlca = True
        if opt == "--aod":
            recoinator.writeAod = True
        if opt == "--global-tag":
            recoinator.globalTag = arg
        if opt == "--lfn" :
            recoinator.inputLFN = arg
        

    recoinator()
