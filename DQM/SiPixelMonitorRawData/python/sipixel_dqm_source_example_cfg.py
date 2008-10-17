import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorRawDataProcess")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")


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

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRZT210_V1.db"
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V6P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/853/62246045-E983-DD11-8742-000423D9863C.root',
        '/store/data/BeamCommissioning08/Cosmics/RAW/v1/000/062/853/C8BD99F1-EA83-DD11-949C-000423D6B48C.root'

    ),
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('debugmessages.txt')
)

process.p1 = cms.Path(process.siPixelDigis*process.SiPixelRawDataErrorSource)
