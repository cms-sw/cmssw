import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *
from CommonTools.ParticleFlow.Isolation.pfMuonIsolationFromDeposits_cff import *

isoDepMuonWithCharged   = isoDepositReplace( 'pfSelectedMuons', 'pfAllChargedHadrons' )
isoDepMuonWithNeutral   = isoDepositReplace( 'pfSelectedMuons', 'pfAllNeutralHadrons' )
isoDepMuonWithPhotons   = isoDepositReplace( 'pfSelectedMuons', 'pfAllPhotons' )
#isoMuonWithElectrons = isoDepositReplace( 'pfSelectedMuons', 'pfAllElectrons' )
#isoMuonWithMuons     = isoDepositReplace( 'pfSelectedMuons', 'pfAllMuons' )

pfMuonIsoDepositsSequence = cms.Sequence(
    isoDepMuonWithCharged   +
    isoDepMuonWithNeutral   +
    isoDepMuonWithPhotons   
#    isoMuonWithElectrons +
#    isoMuonWithMuons
)

pfMuonIsolationSequence = cms.Sequence(
    pfMuonIsoDepositsSequence +
    pfMuonIsolationFromDepositsSequence
    )

