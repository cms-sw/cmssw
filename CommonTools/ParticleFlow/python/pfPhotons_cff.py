import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfSelectedPhotons_cfi import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedPhotons_cfi import *

pfPhotonSequence = cms.Sequence(
    pfSelectedPhotons +
    pfPhotonIsolationSequence +
    # selecting isolated photons:
    pfIsolatedPhotons
    )




