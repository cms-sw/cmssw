import FWCore.ParameterSet.Config as cms

PFRecoTauPFJetInputs = cms.PSet (
    inputJetCollection = cms.InputTag("ak5PFJets"),
    jetConeSize = cms.double(0.5), # for matching between tau and jet
    isolationConeSize = cms.double(0.5), # for the size of the tau isolation
    minJetPt = cms.double(14.0), # do not make taus from jet with pt below that value
    maxJetAbsEta = cms.double(2.5) # do not make taus from jet more forward/backward than this
)
