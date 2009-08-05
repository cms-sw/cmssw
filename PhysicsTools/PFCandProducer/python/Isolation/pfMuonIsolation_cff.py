import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.Isolation.tools_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolationFromDeposits_cff import *

isoMuonWithCharged   = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllChargedHadrons' )
isoMuonWithNeutral   = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllNeutralHadrons' )
isoMuonWithPhotons   = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllPhotons' )
#isoMuonWithElectrons = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllElectrons' )
#isoMuonWithMuons     = isoDepositReplace( 'pfMuonsPtGt5', 'pfAllMuons' )

pfMuonIsoDepositsSequence = cms.Sequence(
    isoMuonWithCharged   +
    isoMuonWithNeutral   +
    isoMuonWithPhotons   
#    isoMuonWithElectrons +
#    isoMuonWithMuons
)

pfMuonIsolationSequence = cms.Sequence(
    pfMuonIsoDepositsSequence +
    pfMuonIsolationFromDepositsSequence
    )

