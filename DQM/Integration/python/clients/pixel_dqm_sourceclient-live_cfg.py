import FWCore.ParameterSet.Config as cms

#from Configuration.StandardSequences.Eras import eras

process = cms.Process("PIXELDQMLIVE")

live=True  #set to false for lxplus offline testing
offlineTesting=not live

TAG ="PixelPhase1" 

process = cms.Process("PIXELDQMLIVE")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis',
                                         'siStripClusters', 
                                         'SiPixelRawDataErrorSource', 
                                         'SiPixelDigiSource'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5

if (live):
    process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
elif(offlineTesting):
    process.load("DQM.Integration.config.fileinputsource_cfi")

#-----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.config.environment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------

process.dqmEnv.subSystemFolder = TAG
process.dqmSaver.tag = TAG

process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/pixel_reference_pp.root'
if (process.runType.getRunType() == process.runType.hi_run):
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/pixel_reference_hi.root'

if (process.runType.getRunType() == process.runType.cosmic_run):
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/pixel_reference_cosmic.root'

#-----------------------------
# Magnetic Field
#-----------------------------

process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# GLOBALTAG
#-------------------------------------------------
# Condition for P5 cluster

if (live):
    process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Condition for lxplus: change and possibly customise the GT
elif(offlineTesting):
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

#-----------------------
#  Reconstruction Modules
#-----------------------

# Real data raw to digi
if (process.runType.getRunType() == process.runType.pp_run): 
    process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

if (process.runType.getRunType() == process.runType.hi_run): 
    process.load("Configuration.StandardSequences.RawToDigi_Repacked_cff")

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_RealData_cfi")

process.siPixelDigis.IncludeErrors = True

process.siPixelDigis.InputLabel   = cms.InputTag("rawDataCollector")
process.siStripDigis.InputLabel   = cms.InputTag("rawDataCollector")

#--------------------------------
# Heavy Ion Configuration Changes
#--------------------------------

if (process.runType.getRunType() == process.runType.hi_run):    
    process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
    process.load('Configuration.StandardSequences.RawToDigi_Repacked_cff')
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    
# Phase1 DQM
process.load("DQM.SiPixelPhase1Config.SiPixelPhase1OnlineDQM_cff")

process.PerModule.enabled=True
process.PerReadout.enabled=True
process.OverlayCurvesForTiming.enabled=False

#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Filters
#--------------------------

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.hltHighLevel.HLTPaths = cms.vstring( 'HLT_ZeroBias_*' , 'HLT_ZeroBias1_*' , 'HLT_PAZeroBias_*' , 'HLT_PAZeroBias1_*', 'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*', 'HLT*SingleMu*','HLT_HICentralityVeto*','HLT_HIMinBias*')
process.hltHighLevel.andOr = cms.bool(True)
process.hltHighLevel.throw =  cms.bool(False)

#--------------------------
# Scheduling
#--------------------------

process.DQMmodules = cms.Sequence(process.dqmEnv*process.dqmSaver)

if (process.runType.getRunType() == process.runType.hi_run):
    process.load("RecoLocalTracker.Configuration.RecoLocalTrackerHeavyIons_cff")
    process.SiPixelPhase1ClustersAnalyzer.pixelSrc = cms.InputTag("siPixelClustersPreSplitting")
#    process.Reco = cms.Sequence(process.siPixelDigis*process.pixeltrackerlocalreco)
    process.Reco =cms.Sequence(process.siPixelDigis*process.siStripDigis*process.siStripVRDigis*process.trackerlocalreco*process.pixeltrackerlocalreco) 
else:
    process.Reco = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.siStripZeroSuppression*process.siStripClusters*process.siPixelClusters)

process.p = cms.Path(
  process.hltHighLevel #trigger selection
 *process.Reco
 *process.DQMmodules
 *process.siPixelPhase1OnlineDQM_source
 *process.siPixelPhase1OnlineDQM_harvesting
)
    
### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#print process.dumpPython()
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()
