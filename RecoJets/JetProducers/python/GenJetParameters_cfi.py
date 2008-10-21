import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: GenJetParameters.cfi,v 1.3 2008/03/11 21:34:33 fedor Exp $
GenJetParameters = cms.PSet(
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(5.0),
    inputEtMin = cms.double(0.0),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)

