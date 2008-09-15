import FWCore.ParameterSet.Config as cms

process = cms.Process("testPixelPopConCalibAnalyzer")
process.load("CondTools.SiPixel.PixelPopConCalibAnalyzer_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.PixelPopConCalibAnalyzer)


