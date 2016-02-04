import FWCore.ParameterSet.Config as cms

hltJetL1Match = cms.EDProducer("HLTJetL1MatchProducer",
    jetsInput = cms.InputTag("hltMCJetCorJetIcone5HF07"),
    L1TauJets = cms.InputTag('hltL1extraParticles','Tau'),
    L1CenJets = cms.InputTag('hltL1extraParticles','Central'),
    L1ForJets = cms.InputTag('hltL1extraParticles','Forward'),
    DeltaR = cms.double(0.5)
)


