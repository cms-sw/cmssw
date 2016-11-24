import FWCore.ParameterSet.Config as cms

process = cms.Process("PIXELDQMDEV")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring( 
                                         ),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

#----------------------------
# Event Source
#-----------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
# dataset /RelValMinBias_13/CMSSW_8_1_0_pre16-81X_upgrade2017_realistic_v22-v1/GEN-SIM-DIGI-RAW
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/0A4D6CFD-E6A6-E611-9078-0CC47A4D7640.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/0C0180C9-D8A6-E611-A1A2-0CC47A7C347A.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/0C168E55-D9A6-E611-88FC-0CC47A4C8E26.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/282EEECB-D8A6-E611-A7F9-0025905A608C.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/3451EF99-DAA6-E611-AD58-0CC47A4C8F08.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/3A97324D-D9A6-E611-8303-0CC47A78A418.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/3CF9A922-D8A6-E611-975B-0CC47A78A4B8.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/404D7F0F-DCA6-E611-91CF-0CC47A4C8E5E.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/4CD952E7-D9A6-E611-B24B-0025905A60E4.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/58F976D4-D8A6-E611-8D80-0025905A6068.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/66E9A0DD-D9A6-E611-90CA-0CC47A4D767E.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/76CD02F8-E6A6-E611-85F6-0CC47A7C35F8.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/7CDABB1E-DBA6-E611-AF0C-0CC47A4C8ECE.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/8E34F8E0-D9A6-E611-A3BE-0CC47A4C8E1E.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/AE364104-DDA6-E611-9C0A-0025905A60D2.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/B2783ACE-D8A6-E611-B6F7-0025905A6134.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/C67FD53A-DEA6-E611-B2DD-0CC47A4D7690.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/D26C799B-DBA6-E611-88F3-0CC47A4C8EA8.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/D4AD584E-D9A6-E611-A7F9-0CC47A7C3444.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/DA860099-DAA6-E611-A5F1-0025905A48C0.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/E651DB20-D8A6-E611-A2CB-0025905A6138.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/F0F86B9F-DBA6-E611-98AC-0CC47A4D760C.root',
       '/store/relval/CMSSW_8_1_0_pre16/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v22-v1/10000/F686FE21-D8A6-E611-9C49-0025905A60C6.root' ] );


secFiles.extend( [
               ] )


#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------
dqmRunConfigDefaults = {
    'userarea': cms.PSet(
        type = cms.untracked.string("userarea"),
        collectorPort = cms.untracked.int32(9190),
        collectorHost = cms.untracked.string('localhost'),
    ),
}

dqmRunConfigType = "userarea"
dqmRunConfig = dqmRunConfigDefaults[dqmRunConfigType]

process.load("DQMServices.Core.DQMStore_cfi")

process.DQM = cms.Service("DQM",
                  debug = cms.untracked.bool(False),
                  publishFrequency = cms.untracked.double(5.0),
                  collectorPort = dqmRunConfig.collectorPort,
                  collectorHost = dqmRunConfig.collectorHost,
                  filter = cms.untracked.string(''),
)

process.DQMMonitoringService = cms.Service("DQMMonitoringService")

process.load("DQMServices.Components.DQMEventInfo_cfi")
process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")

# upload should be either a directory or a symlink for dqm gui destination
process.dqmSaver.path = "." 
process.dqmSaver.producer = 'DQM'
process.dqmSaver.backupLumiCount = 15

TAG = "PixelPhase1"
process.dqmEnv.subSystemFolder = TAG
process.dqmSaver.tag = TAG

#-----------------------------
# Magnetic Field
#-----------------------------
# 3.8T field
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# GLOBALTAG
#-------------------------------------------------
# Condition for P5 cluster
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

#-----------------------
#  Reconstruction Modules
#-----------------------
# Real data raw to digi
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.IncludeErrors = True

# Local Reconstruction
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

#-----------------------
#  Phase1 DQM
#-----------------------

# first, we load the global  defaults and overwrite what needs to be changed
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *
DefaultHisto.enabled = True
DefaultHisto.topFolderName = TAG 

# maximum Lumisection number for trends. This is a hard limit, higher ends up in overflow.
SiPixelPhase1Geometry.max_lumisection = 1000 
# #LS per line in the "overlaid curves"
SiPixelPhase1Geometry.onlineblock = 10 
# number of lines
SiPixelPhase1Geometry.n_onlineblocks = SiPixelPhase1Geometry.max_lumisection.value()/SiPixelPhase1Geometry.onlineblock.value()

# then, we load the online config. This will overwrite more defaults, and e.g. configure for phase0 real data.
process.load("DQM.SiPixelPhase1Config.SiPixelPhase1OnlineDQM_cff")
# this also loads the plugins. After that, some values cannot be changed any more, since they were copied around.

# Now change things back to Phase1 MC
SiPixelPhase1Geometry.n_inner_ring_blades = 22

process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.SiPixelPhase1DigisAnalyzer.src = "simSiPixelDigis"
process.SiPixelPhase1RawDataAnalyzer.src = "simSiPixelDigis"

# All plot configurations should go the the specific config files (for online and offline)
# or to SiPixelPhase1OnlineDQM_cff (if online only). Refer to pixel_up_dqm_sourceclient-live_cfg.py
# to see how things could be overwritten here (works the same in SiPixelPhase1OnlineDQM_cff).

#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Scheduling
#--------------------------
process.DQMmodules = cms.Sequence(process.dqmEnv*process.dqmSaver)

process.p = cms.Path(
    process.siPixelDigis
  * process.siPixelClusters
  * process.DQMmodules
  * process.siPixelPhase1OnlineDQM_source
  * process.siPixelPhase1OnlineDQM_harvesting
)
    
