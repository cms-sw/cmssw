import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *
from CommonTools.ParticleFlow.Isolation.pfElectronIsolationFromDeposits_cff import *
from CommonTools.ParticleFlow.Isolation.electronPFIsolationDeposits_cff import *
from CommonTools.ParticleFlow.Isolation.electronPFIsolationValues_cff import *

isoDepElectronWithCharged   = isoDepositReplace( 'pfSelectedElectrons',
                                              'pfAllChargedHadrons' )
isoDepElectronWithNeutral   = isoDepositReplace( 'pfSelectedElectrons',
                                              'pfAllNeutralHadrons' )
isoDepElectronWithPhotons   = isoDepositReplace( 'pfSelectedElectrons',
                                              'pfAllPhotons' )

pfElectronIsoDepositsSequence = cms.Sequence(
    isoDepElectronWithCharged   +
    isoDepElectronWithNeutral   +
    isoDepElectronWithPhotons   
)

pfElectronIsolationSequence = cms.Sequence(
    pfElectronIsoDepositsSequence +
    pfElectronIsolationFromDepositsSequence +
    electronPFIsolationDepositsSequence +
    electronPFIsolationValuesSequence
    )


