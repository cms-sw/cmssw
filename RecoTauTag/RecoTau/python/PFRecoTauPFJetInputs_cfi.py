import FWCore.ParameterSet.Config as cms

PFRecoTauPFJetInputs = cms.PSet (
    inputJetCollection = cms.InputTag("ak5PFJets"),
    jetConeSize = cms.double(0.5),
    isolationConeSize = cms.double(0.5)
)
