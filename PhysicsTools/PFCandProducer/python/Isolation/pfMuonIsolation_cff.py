import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolationFromDeposits_cff import *

isoDepMuonWithCharged   = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllChargedHadrons' )
isoDepMuonWithNeutral   = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllNeutralHadrons' )
isoDepMuonWithPhotons   = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllPhotons' )
#isoMuonWithElectrons = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllElectrons' )
#isoMuonWithMuons     = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllMuons' )

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

