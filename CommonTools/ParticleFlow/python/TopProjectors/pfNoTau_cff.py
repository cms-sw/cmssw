import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cfi import *

pfNoTauClones = cms.EDProducer("PFJetFromFwdPtrProducer",
                               src = cms.InputTag("pfNoTau"))
