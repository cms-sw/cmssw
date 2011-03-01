import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolationFromDeposits_cff import *


isoDepPhotonWithCharged   = isoDepositReplace( 'pfSelectedPhotons',
                                              'pfAllChargedHadrons' )
isoDepPhotonWithNeutral   = isoDepositReplace( 'pfSelectedPhotons',
                                              'pfAllNeutralHadrons' )
isoDepPhotonWithPhotons   = isoDepositReplace( 'pfSelectedPhotons',
                                              'pfAllPhotons' )
# isoElectronWithElectrons = isoDepositReplace( 'pfSelectedPhotons',
# 'pfAllElectrons' )
# isoElectronWithMuons     = isoDepositReplace( 'pfSelectedPhotons',
#                                               'pfAllMuons' )

pfPhotonIsoDepositsSequence = cms.Sequence(
    isoDepPhotonWithCharged   +
    isoDepPhotonWithNeutral   +
    isoDepPhotonWithPhotons   
#    isoPhotonWithElectrons +
#    isoPhotonWithMuons
)

pfPhotonIsolationSequence = cms.Sequence(
    pfPhotonIsoDepositsSequence +
    pfPhotonIsolationFromDepositsSequence
    )

