import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:Digis_test.root')
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.p1 = cms.Path(process.SiPixelDigiSource)
process.SiPixelDigiSource.saveFile = True
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = True
process.SiPixelDigiSource.ladOn = False
process.SiPixelDigiSource.layOn = False
process.SiPixelDigiSource.phiOn = False
process.SiPixelDigiSource.ringOn = False
process.SiPixelDigiSource.bladeOn = False
process.SiPixelDigiSource.diskOn = False
process.DQM.collectorHost = ''

