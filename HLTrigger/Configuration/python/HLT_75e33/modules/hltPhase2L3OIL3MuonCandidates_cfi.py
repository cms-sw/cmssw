import FWCore.ParameterSet.Config as cms

hltPhase2L3OIL3MuonCandidates = cms.EDProducer("L3MuonCandidateProducer",
    InputLinksObjects = cms.InputTag("hltPhase2L3OIL3MuonsLinksCombination"),
    InputObjects = cms.InputTag("hltPhase2L3OIL3Muons"),
    MuonPtOption = cms.string('Tracker')
)
