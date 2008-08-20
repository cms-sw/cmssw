import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: GenJetParameters_cfi.py,v 1.2 2008/04/21 03:28:21 rpw Exp $
GenJetParameters = cms.PSet(
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(15.0),
    inputEtMin = cms.double(0.0),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)

