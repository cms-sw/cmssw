import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.pfSelectedPhotons_cfi import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolation_cff import *
from CommonTools.ParticleFlow.Isolation.pfIsolatedPhotons_cfi import *

pfPhotonTask = cms.Task(
    pfSelectedPhotons ,
    pfPhotonIsolationTask ,
    # selecting isolated photons:
    pfIsolatedPhotons
    )
pfPhotonSequence = cms.Sequence(pfPhotonTask)
# foo bar baz
# 0H6PRIitrSxkz
# K8nOXMXnwbCL0
