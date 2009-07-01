import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.electronIsolatorFromDeposits_cfi import *


isoElectronWithCharged   = isoDepositReplace( 'pfElectronsPtGt5', 'allChargedHadrons' )
isoElectronWithNeutral   = isoDepositReplace( 'pfElectronsPtGt5', 'allNeutralHadrons' )
isoElectronWithPhotons   = isoDepositReplace( 'pfElectronsPtGt5', 'allPhotons' )
isoElectronWithElectrons = isoDepositReplace( 'pfElectronsPtGt5', 'allElectrons' )
isoElectronWithMuons     = isoDepositReplace( 'pfElectronsPtGt5', 'allMuons' )

electronIsoDepositsSequence = cms.Sequence(
    isoElectronWithCharged   +
    isoElectronWithNeutral   +
    isoElectronWithPhotons   
#    isoElectronWithElectrons +
#    isoElectronWithMuons
)

electronIsolationSequence = cms.Sequence(
    electronIsoDepositsSequence +
    electronIsolatorFromDeposits
    )

