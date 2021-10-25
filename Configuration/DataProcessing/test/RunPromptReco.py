#!/usr/bin/env python3
"""
_RunPromptReco_
Test wrapper to generate a reco config and actually push it into cmsRun for
testing with a few input files etc from the command line
"""
from __future__ import print_function

import sys
import getopt
import traceback
import pickle

from Configuration.DataProcessing.GetScenario import getScenario


class RunPromptReco:

    def __init__(self):
        self.scenario = None
        self.writeRECO = False
        self.writeAOD = False
        self.writeMINIAOD = False
        self.writeDQM = False
        self.writeDQMIO = False
        self.noOutput = False
        self.globalTag = None
        self.inputLFN = None
        self.alcaRecos = None
        self.PhysicsSkims = None
        self.dqmSeq = None
        self.setRepacked = False
        self.isRepacked = False
        self.nThreads = None

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
            raise RuntimeError(msg)
        if self.globalTag == None:
            msg = "No --global-tag specified"
            raise RuntimeError(msg)
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError(msg)

        try:
            scenario = getScenario(self.scenario)
        except Exception as ex:
            msg = "Error getting Scenario implementation for %s\n" % (
                self.scenario,)
            msg += str(ex)
            raise RuntimeError(msg)

        print("Retrieved Scenario: %s" % self.scenario)
        print("Using Global Tag: %s" % self.globalTag)

        dataTiers = []
        if self.writeRECO:
            dataTiers.append("RECO")
            print("Configuring to Write out RECO")
        if self.writeAOD:
            dataTiers.append("AOD")
            print("Configuring to Write out AOD")
        if self.writeMINIAOD:
            dataTiers.append("MINIAOD")
            print("Configuring to Write out MiniAOD")
        if self.writeDQM:
            dataTiers.append("DQM")
            print("Configuring to Write out DQM")
        if self.writeDQMIO:
            dataTiers.append("DQMIO")
            print("Configuring to Write out DQMIO")
        if self.alcaRecos:
            dataTiers.append("ALCARECO")
            print("Configuring to Write out ALCARECO")

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

                if self.PhysicsSkims:
                    kwds['PhysicsSkims'] = self.PhysicsSkims

                if self.dqmSeq:
                    kwds['dqmSeq'] = self.dqmSeq

                if self.setRepacked:
                    kwds['repacked'] = self.isRepacked
                
                if self.nThreads:
                    kwds['nThreads'] = self.nThreads

            process = scenario.promptReco(self.globalTag, **kwds)

        except NotImplementedError as ex:
            print("This scenario does not support Prompt Reco:\n")
            return
        except Exception as ex:
            msg = "Error creating Prompt Reco config:\n"
            msg += traceback.format_exc()
            raise RuntimeError(msg)

        process.source.fileNames.append(self.inputLFN)

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        pklFile = open("RunPromptRecoCfg.pkl", "wb")
        psetFile = open("RunPromptRecoCfg.py", "w")
        try:
            pickle.dump(process, pklFile, protocol=0)
            psetFile.write("import FWCore.ParameterSet.Config as cms\n")
            psetFile.write("import pickle\n")
            psetFile.write("handle = open('RunPromptRecoCfg.pkl','rb')\n")
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

        cmsRun = "cmsRun -e RunPromptRecoCfg.py"
        print("Now do:\n%s" % cmsRun)



if __name__ == '__main__':
    valid = ["scenario=", "reco", "aod", "miniaod","dqm", "dqmio", "no-output", "nThreads=", 
             "global-tag=", "lfn=", "alcarecos=", "PhysicsSkims=", "dqmSeq=", "isRepacked", "isNotRepacked" ]
    usage = \
"""
RunPromptReco.py <options>
Where options are:
 --scenario=ScenarioName
 --reco (to enable RECO output)
 --aod (to enable AOD output)
 --miniaod (to enable MiniAOD output)
 --dqm (to enable DQM output)
 --dqmio (to enable DQMIO output)
 --isRepacked --isNotRepacked (to override default repacked flags)
 --no-output (create config with no output, overrides other settings)
 --global-tag=GlobalTag
 --lfn=/store/input/lfn
 --alcarecos=alcareco_plus_seprated_list
 --PhysicsSkims=skim_plus_seprated_list
 --dqmSeq=dqmSeq_plus_separated_list
 --nThreads=Number_of_cores_or_Threads_used
Example:
python RunPromptReco.py --scenario=cosmics --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever --alcarecos=TkAlCosmics0T+MuAlGlobalCosmics
python RunPromptReco.py --scenario=pp --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever --alcarecos=TkAlMinBias+SiStripCalMinBias
python RunPromptReco.py --scenario=ppEra_Run2_2016 --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever --alcarecos=TkAlMinBias+SiStripCalMinBias --PhysicsSkims=@SingleMuon
"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError as ex:
        print(usage)
        print(str(ex))
        sys.exit(1)


    recoinator = RunPromptReco()

    for opt, arg in opts:
        if opt == "--scenario":
            recoinator.scenario = arg
        if opt == "--reco":
            recoinator.writeRECO = True
        if opt == "--aod":
            recoinator.writeAOD = True
        if opt == "--miniaod":
            recoinator.writeMINIAOD = True
        if opt == "--dqm":
            recoinator.writeDQM = True
        if opt == "--dqmio":
            recoinator.writeDQMIO = True
        if opt == "--no-output":
            recoinator.noOutput = True
        if opt == "--nThreads":
            recoinator.nThreads = arg
        if opt == "--global-tag":
            recoinator.globalTag = arg
        if opt == "--lfn" :
            recoinator.inputLFN = arg
        if opt == "--alcarecos":
            recoinator.alcaRecos = [ x for x in arg.split('+') if len(x) > 0 ]
        if opt == "--PhysicsSkims":
            recoinator.PhysicsSkims = [ x for x in arg.split('+') if len(x) > 0 ]
        if opt == "--dqmSeq":
            recoinator.dqmSeq = [ x for x in arg.split('+') if len(x) > 0 ]
        if opt == "--isRepacked":
            recoinator.setRepacked = True
            recoinator.isRepacked = True
        if opt == "--isNotRepacked":
            recoinator.setRepacked = True
            recoinator.isRepacked = False

    recoinator()
