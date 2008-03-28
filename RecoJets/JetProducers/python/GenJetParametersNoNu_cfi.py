import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: GenJetParametersNoNu.cfi,v 1.1 2007/02/16 23:25:08 fedor Exp $
GenJetParametersNoNu = cms.PSet(
    src = cms.InputTag("genParticlesAllStableNoNu"),
    jetType = cms.untracked.string('GenJet'),
    verbose = cms.untracked.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0)
)

