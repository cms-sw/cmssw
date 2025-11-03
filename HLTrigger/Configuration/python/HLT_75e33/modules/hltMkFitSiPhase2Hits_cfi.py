import FWCore.ParameterSet.Config as cms

# MkFitSiPhase2Hits options
hltMkFitSiPhase2Hits = cms.EDProducer("MkFitPhase2HitConverter",
        mightGet = cms.optional.untracked.vstring,
        hits = cms.InputTag("hltSiPhase2RecHits"),
        clusters = cms.InputTag("hltSiPhase2Clusters"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)
