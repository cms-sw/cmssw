import FWCore.ParameterSet.Config as cms

ticlSeedingL1 = cms.EDProducer("TICLSeedingRegionProducer",
    seedingPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        l1GTCandColl = cms.InputTag("l1tGTProducer", "CL2Photons"),
        maxAbsEta = cms.double(4.0),
        minAbsEta = cms.double(1.3),
        minPt = cms.double(5.0),
        quality = cms.int32(0b0100),
        qualityIsMask = cms.bool(True),
        applyQuality = cms.bool(True),
        type = cms.string('SeedingRegionByL1')
    )
)
