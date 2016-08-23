import FWCore.ParameterSet.Config as cms 

JetHTJetPlusHOFilterSkim = cms.EDFilter("JetHTJetPlusHOFilter",
    Photons = cms.InputTag("photons"),
    PFJets = cms.InputTag("ak4PFJets","","RECO"),
    particleFlowClusterHO = cms.InputTag("particleFlowClusterHO","","RECO"),
    Ptcut = cms.untracked.double(200.0), 
    Etacut = cms.untracked.double(1.5),
    HOcut = cms.untracked.double(8.0)
)

JetHTJetPlusHOFilterSequence = cms.Sequence(JetHTJetPlusHOFilterSkim)
