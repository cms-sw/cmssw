import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolationFromDepositsChargedHadrons_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolationFromDepositsNeutralHadrons_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolationFromDepositsPhotons_cfi import *

# compute isolation values, from various collections of iso deposits.
# could add the electrons and the muons as isolators, just to check that everything is fine

pfElectronIsolationFromDepositsSequence = cms.Sequence(
    pfElectronIsolationFromDepositsChargedHadrons  
#    pfElectronIsolationFromDepositsNeutralHadrons  +
#    pfElectronIsolationFromDepositsPhotons
)



