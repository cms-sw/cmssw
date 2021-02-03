import FWCore.ParameterSet.Config as cms

pfTICL = cms.EDProducer("PFTICLProducer",
    mightGet = cms.optional.untracked.vstring,
    ticlCandidateSrc = cms.InputTag("ticlTrackstersMerge")
)
