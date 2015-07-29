import FWCore.ParameterSet.Config as cms

process = cms.Process("PIXELDQMLIVE")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis', 
                                         #'siPixelClusters', 
                                         'SiPixelRawDataErrorSource', 
                                         'SiPixelDigiSource', 
					 'sipixelEDAClient'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

QTestfile = 'DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml'
#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

##
#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder    = "Pixel"
process.dqmSaver.tag = "Pixel"

process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/pixel_reference_pp.root'
if (process.runType.getRunType() == process.runType.hi_run):
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/pixel_reference_hi.root'

if (process.runType.getRunType() == process.runType.cosmic_run):
    process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/pixel_reference_cosmic.root'
    process.source.SelectEvents = cms.untracked.vstring('HLT*SingleMu*')

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
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

#-----------------------
#  Reconstruction Modules
#-----------------------
# Real data raw to digi
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.IncludeErrors = True

# Local Reconstruction
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

#----------------------------------
# High Pileup Configuration Changes
#----------------------------------
#if (process.runType.getRunType() == process.runType.hpu_run):
#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('HLT_600Tower*','HLT_L1*','HLT_Jet*','HLT_*Cosmic*','HLT_HT*','HLT_MinBias_*','HLT_Physics*', 'HLT_ZeroBias*','HLT_HcalNZS*'))


process.siPixelDigis.InputLabel   = cms.InputTag("rawDataCollector")
#--------------------------------
# Heavy Ion Configuration Changes
#--------------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    QTestfile = 'DQM/SiPixelMonitorClient/test/sipixel_tier0_qualitytest_heavyions.xml'
    process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
    process.load('Configuration.StandardSequences.RawToDigi_Repacked_cff')
    process.siPixelDigis.InputLabel   = cms.InputTag("rawDataRepacker")
#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('HLT_HI*'))

#--------------------------
# Pixel DQM Source and Client
#--------------------------
process.load("DQM.SiPixelCommon.SiPixelP5DQM_source_cff")
process.load("DQM.SiPixelCommon.SiPixelP5DQM_client_cff")

process.qTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath(QTestfile),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    verboseQT = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.sipixelEDAClientP5.inputSource = cms.untracked.string("rawDataCollector")
process.sipixelDaqInfo.daqSource   = cms.untracked.string("rawDataCollector")
process.SiPixelRawDataErrorSource.inputSource  = cms.untracked.string("rawDataCollector")
if (process.runType.getRunType() == process.runType.hi_run):
        process.sipixelEDAClientP5.inputSource = cms.untracked.string("rawDataRepacker")
        process.sipixelDaqInfo.daqSource   = cms.untracked.string("rawDataRepacker")
        process.SiPixelRawDataErrorSource.inputSource  = cms.untracked.string("rawDataRepacker")
#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Filters
#--------------------------
# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

#--------------------------
# Scheduling
#--------------------------
process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters)
process.DQMmodules = cms.Sequence(process.dqmEnv*process.qTester*process.dqmSaver)

process.SiPixelDigiSource.layOn = True
process.SiPixelDigiSource.diskOn = True

process.p = cms.Path(process.Reco*process.DQMmodules*process.SiPixelRawDataErrorSource*process.SiPixelDigiSource*process.SiPixelClusterSource*process.PixelP5DQMClientWithDataCertification)

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()

