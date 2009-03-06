import FWCore.ParameterSet.Config as cms

# Standard PFlowJets parameters
# Joanna Weng
PFJetParameters = cms.PSet(
    src = cms.InputTag("particleFlow"),
    jetType = cms.untracked.string('PFJet'),
    verbose = cms.untracked.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.5)
)

