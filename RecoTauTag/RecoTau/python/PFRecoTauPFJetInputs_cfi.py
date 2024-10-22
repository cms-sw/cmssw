import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

PFRecoTauPFJetInputs = cms.PSet (
    inputJetCollection = cms.InputTag("ak4PFJets"),
    jetConeSize = cms.double(0.5), # for matching between tau and jet
    isolationConeSize = cms.double(0.5), # for the size of the tau isolation
    minJetPt = cms.double(14.0), # do not make taus from jet with pt below that value
    maxJetAbsEta = cms.double(2.7) # do not make taus from jet more forward/backward than this
)
phase2_common.toModify(PFRecoTauPFJetInputs, maxJetAbsEta = cms.double(4.0))
