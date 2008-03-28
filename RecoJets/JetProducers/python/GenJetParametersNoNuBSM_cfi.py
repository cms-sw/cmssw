import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: GenJetParametersNoNuBSM.cfi,v 1.1 2007/05/17 23:56:45 fedor Exp $
GenJetParametersNoNuBSM = cms.PSet(
    src = cms.InputTag("genParticlesAllStableNoNuBSM"),
    jetType = cms.untracked.string('GenJet'),
    verbose = cms.untracked.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0)
)

