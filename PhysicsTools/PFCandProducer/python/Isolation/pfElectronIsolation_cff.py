import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolationFromDeposits_cff import *


isoDepElectronWithCharged   = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllChargedHadrons' )
isoDepElectronWithNeutral   = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllNeutralHadrons' )
isoDepElectronWithPhotons   = isoDepositReplace( 'pfElectronsPtGt5',
                                              'pfAllPhotons' )
# isoElectronWithElectrons = isoDepositReplace( 'pfElectronsPtGt5',
# 'pfAllElectrons' )
# isoElectronWithMuons     = isoDepositReplace( 'pfElectronsPtGt5',
#                                               'pfAllMuons' )

pfElectronIsoDepositsSequence = cms.Sequence(
    isoDepElectronWithCharged   +
    isoDepElectronWithNeutral   +
    isoDepElectronWithPhotons   
#    isoElectronWithElectrons +
#    isoElectronWithMuons
)

pfElectronIsolationSequence = cms.Sequence(
    pfElectronIsoDepositsSequence +
    pfElectronIsolationFromDepositsSequence
    )

