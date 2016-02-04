import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolationFromDeposits_cff import *


isoDepElectronWithCharged   = isoDepositReplace( 'pfSelectedElectrons',
                                              'pfAllChargedHadrons' )
isoDepElectronWithNeutral   = isoDepositReplace( 'pfSelectedElectrons',
                                              'pfAllNeutralHadrons' )
isoDepElectronWithPhotons   = isoDepositReplace( 'pfSelectedElectrons',
                                              'pfAllPhotons' )
# isoElectronWithElectrons = isoDepositReplace( 'pfSelectedElectrons',
# 'pfAllElectrons' )
# isoElectronWithMuons     = isoDepositReplace( 'pfSelectedElectrons',
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

