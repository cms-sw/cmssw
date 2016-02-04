import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.isoValMuonWithCharged_cfi import *
from PhysicsTools.PFCandProducer.Isolation.isoValMuonWithNeutral_cfi import *
from PhysicsTools.PFCandProducer.Isolation.isoValMuonWithPhotons_cfi import *

# compute isolation values, from various collections of iso deposits.
# could add the electrons and the muons as isolators, just to check that everything is fine

pfMuonIsolationFromDepositsSequence = cms.Sequence(
    isoValMuonWithCharged  +
    isoValMuonWithNeutral  +
    isoValMuonWithPhotons
)


