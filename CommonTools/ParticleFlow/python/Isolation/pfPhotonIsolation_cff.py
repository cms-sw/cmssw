import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolationFromDeposits_cff import *
from CommonTools.ParticleFlow.Isolation.photonPFIsolationDeposits_cff import *
from CommonTools.ParticleFlow.Isolation.photonPFIsolationValues_cff import *

isoDepPhotonWithCharged   = isoDepositReplace( 'pfSelectedPhotons',
                                              'pfAllChargedHadrons' )
isoDepPhotonWithNeutral   = isoDepositReplace( 'pfSelectedPhotons',
                                              'pfAllNeutralHadrons' )
isoDepPhotonWithPhotons   = isoDepositReplace( 'pfSelectedPhotons',
                                              'pfAllPhotons' )

pfPhotonIsoDepositsSequence = cms.Sequence(
    isoDepPhotonWithCharged   +
    isoDepPhotonWithNeutral   +
    isoDepPhotonWithPhotons   
)

pfPhotonIsolationSequence = cms.Sequence(
    pfPhotonIsoDepositsSequence +
    pfPhotonIsolationFromDepositsSequence +
    photonPFIsolationDepositsSequence +
    photonPFIsolationValuesSequence
    )

