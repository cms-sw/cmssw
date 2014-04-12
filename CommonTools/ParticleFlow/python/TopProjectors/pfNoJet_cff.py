import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import *
pfNoJetClones = cms.EDProducer(
    "PFCandidateFromFwdPtrProducer",
    src = cms.InputTag("pfNoJet")
)
