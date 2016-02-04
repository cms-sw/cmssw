import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolationFromDeposits_cff import *

pfSelectedPhotons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("pdgId()==22 && mva_nothing_gamma>0")
#    cut = cms.string("pdgId()==22")
)

pfBasedPhotonIsoSequence = cms.Sequence(
    pfParticleSelectionSequence +
    pfSelectedPhotons +
    pfPhotonIsolationSequence+
    pfPhotonIsolationFromDepositsSequence
    ) 
