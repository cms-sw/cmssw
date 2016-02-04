import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.isoValElectronWithCharged_cfi import *
from CommonTools.ParticleFlow.Isolation.isoValElectronWithNeutral_cfi import *
from CommonTools.ParticleFlow.Isolation.isoValElectronWithPhotons_cfi import *

# compute isolation values, from various collections of iso deposits.
# could add the electrons and the muons as isolators, just to check that everything is fine

pfElectronIsolationFromDepositsSequence = cms.Sequence(
    isoValElectronWithCharged  +
    isoValElectronWithNeutral  +
    isoValElectronWithPhotons
)



