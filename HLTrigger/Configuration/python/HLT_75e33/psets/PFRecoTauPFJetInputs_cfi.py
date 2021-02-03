import FWCore.ParameterSet.Config as cms

PFRecoTauPFJetInputs = cms.PSet(
    inputJetCollection = cms.InputTag("ak4PFJets"),
    isolationConeSize = cms.double(0.5),
    jetConeSize = cms.double(0.5),
    maxJetAbsEta = cms.double(4.0),
    minJetPt = cms.double(14.0)
)