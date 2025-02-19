import FWCore.ParameterSet.Config as cms

fixedGridRhoCentral = cms.EDProducer("FixedGridRhoProducer",
    pfCandidatesTag = cms.InputTag("particleFlow"),
    EtaRegion = cms.string("Central")
)

fixedGridRhoForward = cms.EDProducer("FixedGridRhoProducer",
    pfCandidatesTag = cms.InputTag("particleFlow"),
    EtaRegion = cms.string("Forward")
)

fixedGridRhoAll = cms.EDProducer("FixedGridRhoProducer",
    pfCandidatesTag = cms.InputTag("particleFlow"),
    EtaRegion = cms.string("All")
)


