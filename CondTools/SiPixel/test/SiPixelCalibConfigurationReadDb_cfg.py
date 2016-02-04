import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelCalibConfRead")
process.load("CondTools.SiPixel.SiPixelCalibConfiguration_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
                            timetype=cms.string("runnumber"),
                            firstValue = cms.uint64(63033),
                            lastValue = cms.uint64(63033),
                            interval = cms.uint64(1)
                            )

process.readcalib = cms.EDAnalyzer("SiPixelCalibConfigurationReadDb",
                                 verbosity=cms.bool(False)
                                 )

process.p1 = cms.Path(process.readcalib)
process.sipixelcalib_essource.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelCalibConfigurationRcd'),
    tag = cms.string('SiPixelCalibConfiguration_test')
#    tag = cms.string('PixelAlive_default')
#    tag = cms.string('SCurve_default')

))
process.sipixelcalib_essource.connect='sqlite_file:siPixelCalibConfiguration.db'


