import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolationFromDepositsChargedHadrons_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolationFromDepositsNeutralHadrons_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolationFromDepositsPhotons_cfi import *

# compute isolation values, from various collections of iso deposits.
# could add the electrons and the muons as isolators, just to check that everything is fine

pfMuonIsolationFromDepositsSequence = cms.Sequence(
    pfMuonIsolationFromDepositsChargedHadrons  +
    pfMuonIsolationFromDepositsNeutralHadrons  +
    pfMuonIsolationFromDepositsPhotons
)


