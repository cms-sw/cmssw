import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.isoValPhotonWithCharged_cfi import *
from CommonTools.ParticleFlow.Isolation.isoValPhotonWithNeutral_cfi import *
from CommonTools.ParticleFlow.Isolation.isoValPhotonWithPhotons_cfi import *

# compute isolation values, from various collections of iso deposits.
# could add the electrons and the muons as isolators, just to check that everything is fine

pfPhotonIsolationFromDepositsSequence = cms.Sequence(
    isoValPhotonWithCharged  +
    isoValPhotonWithNeutral  +
    isoValPhotonWithPhotons
)



