import FWCore.ParameterSet.Config as cms

PFRecoTauPFJetInputs = cms.PSet (
    inputJetCollection = cms.InputTag("ak5PFJets"),
    jetConeSize = cms.double(0.5), # for matching between tau and jet
    isolationConeSize = cms.double(0.5) # for the size of the tau isolation
)
