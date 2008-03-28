import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetParameters.cfi,v 1.2 2008/03/11 21:34:33 fedor Exp $
CaloJetParameters = cms.PSet(
    src = cms.InputTag("caloTowers"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

