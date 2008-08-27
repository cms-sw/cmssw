import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/555/02E4041E-1571-DD11-98CE-001D09F241B9.root')
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.p1 = cms.Path(process.SiPixelDigiSource)
process.SiPixelDigiSource.saveFile = True
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = False
process.SiPixelDigiSource.ladOn = True
process.SiPixelDigiSource.layOn = True
process.SiPixelDigiSource.phiOn = True
process.SiPixelDigiSource.ringOn = True
process.SiPixelDigiSource.bladeOn = True
process.SiPixelDigiSource.diskOn = True
process.DQM.collectorHost = ''

