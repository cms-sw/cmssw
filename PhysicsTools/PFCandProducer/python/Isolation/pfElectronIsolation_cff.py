import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolationFromDeposits_cff import *


isoElectronWithCharged   = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllChargedHadrons' )
isoElectronWithNeutral   = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllNeutralHadrons' )
isoElectronWithPhotons   = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllPhotons' )
isoElectronWithElectrons = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllElectrons' )
isoElectronWithMuons     = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllMuons' )

pfElectronIsoDepositsSequence = cms.Sequence(
    isoElectronWithCharged   
#    isoElectronWithNeutral   +
#    isoElectronWithPhotons   
#    isoElectronWithElectrons +
#    isoElectronWithMuons
)

pfElectronIsolationSequence = cms.Sequence(
    pfElectronIsoDepositsSequence +
    pfElectronIsolationFromDepositsSequence
    )

