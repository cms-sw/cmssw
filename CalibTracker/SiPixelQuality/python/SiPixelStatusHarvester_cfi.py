import FWCore.ParameterSet.Config as cms

siPixelStatusHarvester = cms.EDAnalyzer("SiPixelStatusHarvester",

    SiPixelStatusManagerParameters = cms.PSet(
        outputBase = cms.untracked.string("runbased"), #nLumibased #runbased #dynamicLumibased
        aveDigiOcc = cms.untracked.int32(20000),
        resetEveryNLumi = cms.untracked.int32(10),
        moduleName = cms.untracked.string("siPixelStatusProducer"),
        label      = cms.untracked.string("siPixelStatus"),
    ),
    debug = cms.untracked.bool(False),
    recordName   = cms.untracked.string("SiPixelQualityFromDbRcd")

)

