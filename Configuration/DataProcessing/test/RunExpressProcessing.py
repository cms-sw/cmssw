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
        self.writeRaw = False
        self.writeReco = False
        self.writeFevt = False
        self.writeAlca = False
        self.writeDqm = False
        self.noOutput = False
        self.globalTag = None
        self.inputLFN = None
        self.alcaRecos = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError, msg
        if self.globalTag == None:
            msg = "No --global-tag specified"
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
        if self.writeRaw:
            dataTiers.append("RAW")
            print "Configuring to Write out Raw..."
        if self.writeReco:
            dataTiers.append("RECO")
            print "Configuring to Write out Reco..."
        if self.writeFevt:
            dataTiers.append("FEVT")
            print "Configuring to Write out Fevt..."
        if self.writeAlca:
            dataTiers.append("ALCARECO")
            print "Configuring to Write out Alca..."
        if self.writeDqm:
            dataTiers.append("DQM")
            print "Configuring to Write out Dqm..."

        try:
            kwds = {}

            if self.noOutput:
                # get config without any output
                kwds['writeTiers'] = []

            elif len(dataTiers) > 0:
                # get config with specified output
                kwds['writeTiers'] = dataTiers

            # if none of the above use default output data tiers


            if not self.alcaRecos is None:
                # if skims specified from command line than overwrite the defaults
                kwds['skims'] = self.alcaRecos


            process = scenario.expressProcessing(self.globalTag, **kwds)

        except NotImplementedError, ex:
            print "This scenario does not support Express Processing:\n"
            return
        except Exception, ex:
            msg = "Error creating Express Processing config:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames = [self.inputLFN]

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        psetFile = open("RunExpressProcessingCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunExpressProcessingCfg.py"
        print "Now do:\n%s" % cmsRun



if __name__ == '__main__':
    valid = ["scenario=", "raw", "reco", "fevt", "alca", "dqm", "no-output",
             "global-tag=", "lfn=", 'alcaRecos=']
    usage = \
"""
RunExpressProcessing.py <options>

Where options are:
 --scenario=ScenarioName
 --raw (to enable RAW output)
 --reco (to enable RECO output)
 --fevt (to enable FEVT output)
 --alca (to enable ALCARECO output)
 --dqm (to enable DQM output)
 --no-output (create config with no output, overrides other settings)
 --global-tag=GlobalTag
 --lfn=/store/input/lfn
 --alcaRecos=commasepratedlist

Example:
python RunExpressProcessing.py --scenario cosmics --global-tag GLOBALTAG::ALL --lfn /store/whatever --fevt --alca --dqm

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
        if opt == "--raw":
            expressinator.writeRaw = True
        if opt == "--reco":
            expressinator.writeReco = True
        if opt == "--fevt":
            expressinator.writeFevt = True
        if opt == "--alca":
            expressinator.writeAlca = True
        if opt == "--dqm":
            expressinator.writeDqm = True
        if opt == "--no-output":
            expressinator.noOutput = True
        if opt == "--global-tag":
            expressinator.globalTag = arg
        if opt == "--lfn" :
            expressinator.inputLFN = arg
        if opt == "--alcaRecos":
            expressinator.alcaRecos = [ x for x in arg.split('+') if len(x) > 0 ]

    expressinator()
