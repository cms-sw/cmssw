import FWCore.ParameterSet.Config as cms

process = cms.Process("SimAnalyzer")
#Geometry
#
process.load("Geometry.CMSCommonData.cmsRecoIdealGeometryXML_cfi")

#Magnetic Field
#
process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:LaserEvents.SIM-DIGI.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.ana = cms.EDAnalyzer("SimAnalyzer",
    SearchWindowPhiTEC = cms.untracked.double(0.05),
    SearchWindowPhiTIB = cms.untracked.double(0.05),
    SearchWindowZTOB = cms.untracked.double(1.0),
    ROOTFileCompression = cms.untracked.int32(1),
    ROOTFileName = cms.untracked.string('simulation.histos.root'),
    SearchWindowPhiTOB = cms.untracked.double(0.05),
    SearchWindowZTIB = cms.untracked.double(1.0),
    DebugLevel = cms.untracked.int32(3)
)

process.p1 = cms.Path(process.ana)

