import FWCore.ParameterSet.Config as cms

hltFixedGridRhoFastjetAllCaloForEGamma = cms.EDProducer("FixedGridRhoProducerFastjet",
    gridSpacing = cms.double(0.55),
    maxRapidity = cms.double(2.5),
    pfCandidatesTag = cms.InputTag("hltTowerMakerForAllForEgamma")
)
