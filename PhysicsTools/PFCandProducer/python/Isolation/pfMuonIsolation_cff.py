import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolationFromDeposits_cff import *

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

