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
        self.writeRAW = False
        self.writeRECO = False
        self.writeFEVT = False
        self.writeDQM = False
        self.writeDQMIO = False
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
        if self.writeRAW:
            dataTiers.append("RAW")
            print "Configuring to Write out RAW"
        if self.writeRECO:
            dataTiers.append("RECO")
            print "Configuring to Write out RECO"
        if self.writeFEVT:
            dataTiers.append("FEVT")
            print "Configuring to Write out FEVT"
        if self.writeDQM:
            dataTiers.append("DQM")
            print "Configuring to Write out DQM"
        if self.writeDQMIO:
            dataTiers.append("DQMIO")
            print "Configuring to Write out DQMIO"
        if self.alcaRecos:
            dataTiers.append("ALCARECO")
            print "Configuring to Write out ALCARECO"


        try:
            kwds = {}

            if self.noOutput:
                kwds['outputs'] = []
            else:
                outputs = []
                for dataTier in dataTiers:
                    outputs.append({ 'dataTier' : dataTier,
                                     'eventContent' : dataTier,
                                     'moduleLabel' : "write_%s" % dataTier })
                kwds['outputs'] = outputs

                if self.alcaRecos:
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
    valid = ["scenario=", "raw", "reco", "fevt", "dqm", "dqmio", "no-output",
             "global-tag=", "lfn=", 'alcarecos=']
    usage = \
"""
RunExpressProcessing.py <options>

Where options are:
 --scenario=ScenarioName
 --raw (to enable RAW output)
 --reco (to enable RECO output)
 --fevt (to enable FEVT output)
 --dqm (to enable DQM output)
 --no-output (create config with no output, overrides other settings)
 --global-tag=GlobalTag
 --lfn=/store/input/lfn
 --alcarecos=plus_seprated_list

Examples:

python RunExpressProcessing.py --scenario cosmics --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio --alcarecos=TkAlCosmics0T+SiStripCalZeroBias

python RunExpressProcessing.py --scenario pp --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio --alcarecos=TkAlMinBias+SiStripCalZeroBias

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
            expressinator.writeRAW = True
        if opt == "--reco":
            expressinator.writeRECO = True
        if opt == "--fevt":
            expressinator.writeFEVT = True
        if opt == "--dqm":
            expressinator.writeDQM = True
        if opt == "--dqmio":
            expressinator.writeDQMIO = True
        if opt == "--no-output":
            expressinator.noOutput = True
        if opt == "--global-tag":
            expressinator.globalTag = arg
        if opt == "--lfn" :
            expressinator.inputLFN = arg
        if opt == "--alcarecos":
            expressinator.alcaRecos = [ x for x in arg.split('+') if len(x) > 0 ]

    expressinator()
