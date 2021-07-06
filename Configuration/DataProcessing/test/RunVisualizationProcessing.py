#!/usr/bin/env python3
"""
_RunVisualizationProcessing_

Test wrapper to generate an express processing config and actually push
it into cmsRun for testing with a few input files etc from the command line

"""
from __future__ import print_function

import sys
import getopt
import pickle

from Configuration.DataProcessing.GetScenario import getScenario



class RunVisualizationProcessing:

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
        self.preFilter = None

        
 #FIXME: should add an option to specify an EDM input source?
 

    def __call__(self):
        if self.scenario == None:
            msg = "No --scenario specified"
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
        print("Using Global Tag: %s" % self.globalTag)

        dataTiers = []
        if self.writeRaw:
            dataTiers.append("RAW")
            print("Configuring to Write out Raw...")
        if self.writeReco:
            dataTiers.append("RECO")
            print("Configuring to Write out Reco...")
        if self.writeFevt:
            dataTiers.append("FEVT")
            print("Configuring to Write out Fevt...")
        if self.writeAlca:
            dataTiers.append("ALCARECO")
            print("Configuring to Write out Alca...")
        if self.writeDqm:
            dataTiers.append("DQM")
            print("Configuring to Write out Dqm...")



        try:
            kwds = {}
            if self.inputLFN != None:
                kwds['inputSource'] = 'EDM'
                
            if self.noOutput:
                # get config without any output
                kwds['writeTiers'] = []

            elif len(dataTiers) > 0:
                # get config with specified output
                kwds['writeTiers'] = dataTiers

            if self.preFilter:
                kwds['preFilter'] = self.preFilter


            # if none of the above use default output data tiers

            process = scenario.visualizationProcessing(self.globalTag, **kwds)

        except NotImplementedError as ex:
            print("This scenario does not support Visualization Processing:\n")
            return
        except Exception as ex:
            msg = "Error creating Visualization Processing config:\n"
            msg += str(ex)
            raise RuntimeError(msg)

        if self.inputLFN != None:
            process.source.fileNames = [self.inputLFN]

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        pklFile = open("RunVisualizationProcessingCfg.pkl", "wb")
        psetFile = open("RunVisualizationProcessingCfg.py", "w")
        try:
            pickle.dump(process, pklFile, protocol=0)
            psetFile.write("import FWCore.ParameterSet.Config as cms\n")
            psetFile.write("import pickle\n")
            psetFile.write("handle = open('RunVisualizationProcessingCfg.pkl','rb')\n")
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

        cmsRun = "cmsRun -e RunVisualizationProcessingCfg.py"
        print("Now do:\n%s" % cmsRun)



if __name__ == '__main__':
    valid = ["scenario=", "reco", "fevt", "no-output",
             "global-tag=", "lfn=",'preFilter=']
    usage = \
"""
RunVisualizationProcessing.py <options>

Where options are:
 --scenario=ScenarioName
 --reco (to enable RECO output)
 --fevt (to enable FEVT output)
 --no-output (create config with no output, overrides other settings)
 --global-tag=GlobalTag
 --lfn=/store/input/lfn
 --preFilter=/sybsystem/package/filtername.sequence
 
Example:
python RunVisualizationProcessing.py --scenario cosmics --global-tag GLOBALTAG::ALL --lfn /store/whatever --reco

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError as ex:
        print(usage)
        print(str(ex))
        sys.exit(1)


    visualizator = RunVisualizationProcessing()

    for opt, arg in opts:
        if opt == "--scenario":
            visualizator.scenario = arg
        if opt == "--reco":
            visualizator.writeReco = True
        if opt == "--fevt":
            visualizator.writeFevt = True
        if opt == "--no-output":
            visualizator.noOutput = True
        if opt == "--global-tag":
            visualizator.globalTag = arg
        if opt == "--lfn" :
            visualizator.inputLFN = arg
        if opt == "--preFilter":
            visualizator.preFilter = arg

    visualizator()
