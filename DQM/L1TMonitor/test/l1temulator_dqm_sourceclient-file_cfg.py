#
# cfg file to run online L1 Trigger emulator DQM
#
# V M Ghete 2010-07-09


import FWCore.ParameterSet.Config as cms
import sys

# choose the environment you run
#l1DqmEnv = 'live'
#l1DqmEnv = 'playback'
l1DqmEnv = 'file'

process = cms.Process("L1TEmuDQM")

# check that a valid choice for environment exists

if not ((l1DqmEnv == 'live') or l1DqmEnv == 'playback' or l1DqmEnv == 'file') : 
    print 'No valid input source was chosen. Your value for l1DqmEnv input parameter is:'  
    print 'l1DqmEnv = ', l1DqmEnv
    print 'Available options: "live", "playback", "file" '
    sys.exit()

#----------------------------
# Event Source
#

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.inputsource_cfi")
    process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring("*")
            )
    process.EventStreamHttpReader.consumerName = 'L1TEMU DQM Consumer'
    process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(25.0)
 
elif l1DqmEnv == 'playback' :
    print 'FIXME'
    
else : 
    # running on a file
    process.load("DQM.L1TMonitor.inputsource_file_cfi")
    
      
#----------------------------
# DQM Environment
#

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'L1TEMU'

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.environment_cfi")
    # no references needed

    #
    # load and configure modules via Global Tag
    # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
    process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

elif l1DqmEnv == 'playback' :
    print 'FIXME'
    
else : 
    # running on a file
    process.load("DQM.L1TMonitor.environment_file_cfi")

    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    process.GlobalTag.globaltag = 'GR_R_38X_V9::All'
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')


process.load("Configuration.StandardSequences.Geometry_cff")


#-----------------------------
#
#  L1 Emulator DQM SOURCES
#

process.load("DQM.L1TMonitor.L1TEmulatorMonitor_cff")    
process.load("DQM.L1TMonitorClient.L1TEMUMonitorClient_cff")    

# NL//this over-writting may be employed only when needed
#  ie quick module disabling, before new tags can be corrected)
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
l1compare.COMPARE_COLLS = [1, 1, 1, 1,  1, 1, 1, 1, 1, 0, 1, 1]

newHWSequence = cms.Sequence(
                        deEcal+
                        deHcal+
                        deRct+
                        deGct+
                        deDt+
                        deDttf+
                        deCsc+
                        deCsctf+
                        deRpc+
                        deGmt+
                        deGt*
                        l1compare)
process.globalReplace("L1HardwareValidation", newHWSequence)

#
# fast over-mask a system: if the name of the system is in the list, the system will be masked
# (the default mask value is given in L1Systems VPSet)             
#
# names are case sensitive, order is irrelevant
# "ECAL", "HCAL", "RCT", "GCT", "DTTF", "DTTPG", "CSCTF", "CSCTPG", "RPC", "GMT", "GT"
#
process.l1temuEventInfoClient.MaskL1Systems = cms.vstring()
#
# fast over-mask an object: if the name of the object is in the list, the object will be masked
# (the default mask value is given in L1Objects VPSet)             
#
# names are case sensitive, order is irrelevant
# 
# "Mu", "NoIsoEG", "IsoEG", "CenJet", "ForJet", "TauJet", "ETM", "ETT", "HTT", "HTM", 
# "HfBitCounts", "HfRingEtSums", "TechTrig", "GtExternal
#
process.l1temuEventInfoClient.MaskL1Objects =  cms.vstring()   

# 
#process.l1temuEventInfoClient.verbose = cms.untracked.bool(True)


