import FWCore.ParameterSet.Config as cms 

SinglePhotonJetPlusHOFilterSkim = cms.EDFilter("SinglePhotonJetPlusHOFilter",
    Photons = cms.InputTag("photons"),
    PFJets = cms.InputTag("ak4PFJets","","RECO"),
    particleFlowClusterHO = cms.InputTag("particleFlowClusterHO","","RECO"),
    Ptcut = cms.untracked.double(90.0), 
    Etacut = cms.untracked.double(1.5),
    HOcut = cms.untracked.double(5.0), 
    Pho_Ptcut = cms.untracked.double(120.0)
) 

SinglePhotonJetPlusHOFilterSequence = cms.Sequence(SinglePhotonJetPlusHOFilterSkim)

