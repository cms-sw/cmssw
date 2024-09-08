import FWCore.ParameterSet.Config as cms

hltPhase2OTHitsInputLST = cms.EDProducer('LSTPhase2OTHitsInputProducer',
    phase2OTRecHits = cms.InputTag('hltSiPhase2RecHits'),
    mightGet = cms.optional.untracked.vstring
)

