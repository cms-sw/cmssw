import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.muonIsolatorFromDeposits_cfi import *

isoMuonWithCharged   = isoDepositReplace( 'pfMuonsPtGt5', 'allChargedHadrons' )
isoMuonWithNeutral   = isoDepositReplace( 'pfMuonsPtGt5', 'allNeutralHadrons' )
isoMuonWithPhotons   = isoDepositReplace( 'pfMuonsPtGt5', 'allPhotons' )
isoMuonWithElectrons = isoDepositReplace( 'pfMuonsPtGt5', 'allElectrons' )
isoMuonWithMuons     = isoDepositReplace( 'pfMuonsPtGt5', 'allMuons' )

muonIsoDepositsSequence = cms.Sequence(
    isoMuonWithCharged   +
    isoMuonWithNeutral   +
    isoMuonWithPhotons   +
    isoMuonWithElectrons +
    isoMuonWithMuons
)

muonIsolationSequence = cms.Sequence(
    muonIsoDepositsSequence +
    muonIsolatorFromDeposits
    )

