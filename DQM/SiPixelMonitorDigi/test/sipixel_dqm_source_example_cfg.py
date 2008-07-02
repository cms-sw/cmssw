import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.load("DQMServices.Core.DQM_cfg")

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
process.SiPixelDigiSource.modOn = False
process.SiPixelDigiSource.ladOn = True
process.SiPixelDigiSource.layOn = True
process.SiPixelDigiSource.phiOn = True
process.SiPixelDigiSource.ringOn = True
process.SiPixelDigiSource.bladeOn = True
process.SiPixelDigiSource.diskOn = True
process.DQM.collectorHost = ''

