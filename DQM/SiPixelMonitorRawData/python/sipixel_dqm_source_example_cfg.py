import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorRawDataProcess")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

# Database configuration
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierInt/CMS_COND_30X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_P_V8::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

# Pixel RawToDigi conversion
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = "source"
#  process.siPixelDigis.InputLabel = "siPixelRawData"
process.siPixelDigis.Timing = False
process.siPixelDigis.IncludeErrors = True
#  process.siPixelDigis.CheckPixelOrder = True

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")
process.SiPixelRawDataErrorSource.saveFile = True
#process.SiPixelHLTSource.isPIB = False
process.SiPixelRawDataErrorSource.slowDown = False
#process.SiPixelHLTSource.reducedSet = False
process.SiPixelRawDataErrorSource.modOn = True
process.SiPixelRawDataErrorSource.ladOn = False
process.SiPixelRawDataErrorSource.bladeOn = False

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/853/62246045-E983-DD11-8742-000423D9863C.root',
        '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/853/C8BD99F1-EA83-DD11-949C-000423D6B48C.root'

    ),
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
)

process.p1 = cms.Path(process.siPixelDigis*process.SiPixelRawDataErrorSource)
