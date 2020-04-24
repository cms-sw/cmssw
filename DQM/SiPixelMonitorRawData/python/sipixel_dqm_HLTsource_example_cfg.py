import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

# Database configuration
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# conditions
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR10_P_V4::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

# Pixel RawToDigi conversion
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = "source"
process.siPixelDigis.Timing = False
process.siPixelDigis.IncludeErrors = True

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorHLT_cfi")
process.SiPixelHLTSource.saveFile = True
#process.SiPixelHLTSource.isPIB = False
process.SiPixelHLTSource.slowDown = False
#process.SiPixelHLTSource.reducedSet = False


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
  '/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/877/FAC1761E-A64F-DF11-BD37-003048D2BDD8.root'
#  '/store/data/Commissioning10/MinimumBias/RAW/v4/000/133/877/FADF1B51-BF4F-DF11-9CE2-001D09F24353.root'
    )
)


##
## number of events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) )

process.p1 = cms.Path(process.siPixelDigis*process.SiPixelHLTSource)

process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.cerr.threshold = 'INFO'
