import FWCore.ParameterSet.Config as cms


fixedGridRhoFastjetAll = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("particleFlow"),
    maxRapidity = cms.double(5.0),
    gridSpacing = cms.double(0.55)
)


fixedGridRhoFastjetAllCalo = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("towerMaker"),
    maxRapidity = cms.double(5.0),
    gridSpacing = cms.double(0.55)
)




