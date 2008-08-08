import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRUZET3_V7.db"
process.GlobalTag.globaltag = "CRUZET3_V7::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/chiochia/cmssw/Muon_FullValidation_150pre3.root')
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.p1 = cms.Path(process.SiPixelDigiSource)
process.SiPixelDigiSource.saveFile = True
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = True
process.SiPixelDigiSource.twoDimOn = True
process.SiPixelDigiSource.ladOn = False
process.SiPixelDigiSource.layOn = False
process.SiPixelDigiSource.phiOn = False
process.SiPixelDigiSource.ringOn = False
process.SiPixelDigiSource.bladeOn = False
process.SiPixelDigiSource.diskOn = False
process.DQM.collectorHost = ''

