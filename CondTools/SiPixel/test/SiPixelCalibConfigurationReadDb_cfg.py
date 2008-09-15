import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelCalibConfRead")
process.load("CondTools.SiPixel.SiPixelCalibConfiguration_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.readcalib = cms.EDFilter("SiPixelCalibConfigurationReadDb")

process.p1 = cms.Path(process.readcalib)
process.sipixelcalib_essource.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelCalibConfigurationRcd'),
    tag = cms.string('PixelAlive_281')
))


