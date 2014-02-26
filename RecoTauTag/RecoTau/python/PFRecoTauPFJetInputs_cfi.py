import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.pfJetSelector_cfi import pfJetSelector

tauInputJets = pfJetSelector.clone(
    src = cms.InputTag("ak5PFJets"),
    filter = cms.bool( True ),
    cut = cms.string ( "pt > 20 && abs(eta) < 2.5")
)

PFRecoTauPFJetInputs = cms.PSet (
    inputJetCollection = cms.InputTag("tauInputJets"),
    jetConeSize = cms.double(0.5),
    isolationConeSize = cms.double(0.5)
)
