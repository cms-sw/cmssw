import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.pfJetSelector_cfi import pfJetSelector

tauInputJets = pfJetSelector.clone(
    src = cms.InputTag("ak5PFJets"),
    filter = cms.bool( False ),
    cut = cms.string ( "pt > 14.5 && abs(eta) < 3.0")
)

PFRecoTauPFJetInputs = cms.PSet (
    inputJetCollection = cms.InputTag("tauInputJets"),
    jetConeSize = cms.double(0.5),
    isolationConeSize = cms.double(0.5)
)
