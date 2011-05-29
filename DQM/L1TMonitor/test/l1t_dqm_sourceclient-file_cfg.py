#
# cfg file to run online L1 Trigger DQM
#
# V M Ghete 2010-07-09


import FWCore.ParameterSet.Config as cms
import sys

# choose the environment you run
#l1DqmEnv = 'live'
#l1DqmEnv = 'playback'
l1DqmEnv = 'file'

process = cms.Process("DQM")

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
    process.EventStreamHttpReader.consumerName = 'L1T DQM Consumer'
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
process.dqmEnv.subSystemFolder = 'L1T'

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.environment_cfi")
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

    #
    # load and configure modules via Global Tag
    # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
    process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

elif l1DqmEnv == 'playback' :
    print 'FIXME'
    
elif l1DqmEnv == 'file-P5' :
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
    process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)
    
else : 
    # running on a file, on lxplus (not on .cms)
    process.load("DQM.L1TMonitor.environment_file_cfi")

    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    process.GlobalTag.globaltag = 'GR_P_V20::All'
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')


process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------
# sequences needed for L1 trigger DQM
#

# standard unpacking sequence 
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

# L1 Trigger sequences 

# l1tMonitor and l1tMonitorEndPathSeq
process.load("DQM.L1TMonitor.L1TMonitor_cff")    

# l1tMonitorClient and l1tMonitorClientEndPathSeq
process.load("DQM.L1TMonitorClient.L1TMonitorClient_cff")    

#-------------------------------------
# paths & schedule for L1 Trigger DQM
#

# TODO define a L1 trigger L1TriggerRawToDigi in the standard sequence 
# to avoid all these remove
process.rawToDigiPath = cms.Path(process.RawToDigi)
#
process.RawToDigi.remove("siPixelDigis")
process.RawToDigi.remove("siStripDigis")
process.RawToDigi.remove("scalersRawToDigi")
process.RawToDigi.remove("castorDigis")
# for GCT, unpack all five samples
process.RawToDigi.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

# 
process.l1tMonitorPath = cms.Path(process.l1tMonitor)

#
process.l1tMonitorClientPath = cms.Path(process.l1tMonitorClient)

#
process.l1tMonitorEndPath = cms.EndPath(process.l1tMonitorEndPathSeq)

#
process.l1tMonitorClientEndPath = cms.EndPath(process.l1tMonitorClientEndPathSeq)

#
process.dqmEndPath = cms.EndPath(
                                 process.dqmEnv *
                                 process.dqmSaver
                                 )

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1tMonitorPath,
                                process.l1tMonitorClientPath,
                                process.l1tMonitorEndPath,
                                process.l1tMonitorClientEndPath,
                                process.dqmEndPath
                                )

#---------------------------------------------

# examples for quick fixes in case of troubles 
#    please do not modify the commented lines
#


#
# turn on verbosity in L1TEventInfoClient
#
# process.l1tEventInfoClient.verbose = cms.untracked.bool(True)


#
# available data masks (case insensitive):
#    all, gt, muons, jets, taujets, isoem, nonisoem, met
process.l1tEventInfoClient.dataMaskedSystems = cms.untracked.vstring("Muons","Jets","TauJets","IsoEm","NonIsoEm","MET")

#
# available emulator masks (case insensitive):
#    all, dttf, dttpg, csctf, csctpg, rpc, gmt, ecal, hcal, rct, gct, glt
process.l1tEventInfoClient.emulatorMaskedSystems = cms.untracked.vstring("All")
