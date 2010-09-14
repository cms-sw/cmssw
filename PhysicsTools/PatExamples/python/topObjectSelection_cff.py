import FWCore.ParameterSet.Config as cms

## ---
##
## this cff file keep all object selections used for the TopPAG
## reference selection for ICHEP 2010
##
## ---
from PhysicsTools.PatAlgos.cleaningLayer1.muonCleaner_cfi import *
looseMuons = cleanPatMuons.clone(
    preselection =
    'isGlobalMuon & isTrackerMuon &'
    'pt > 20. &'
    'abs(eta) < 2.1 &'
    '(trackIso+caloIso)/pt < 0.1 &'
    'innerTrack.numberOfValidHits > 10 &'
    'globalTrack.normalizedChi2 < 10.0 &'
    'globalTrack.hitPattern.numberOfValidMuonHits > 0 &'
    'abs(dB) < 0.02',
    checkOverlaps = cms.PSet(
      jets = cms.PSet(
        src                 = cms.InputTag("goodJets"),
        algorithm           = cms.string("byDeltaR"),
        preselection        = cms.string(""),
        deltaR              = cms.double(0.3),
        checkRecoComponents = cms.bool(False),
        pairCut             = cms.string(""),
        requireNoOverlaps   = cms.bool(True),
      )
    )
)

tightMuons = cleanPatMuons.clone(
    src = 'looseMuons',
    preselection = '(trackIso+caloIso)/pt < 0.05'
)

vetoMuons = cleanPatMuons.clone(
    preselection =
    'isGlobalMuon &'
    'pt > 10. &'
    'abs(eta) < 2.5 &'
    '(trackIso+caloIso)/pt < 0.2'
)

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
vetoElecs = selectedPatElectrons.clone(
    src = 'selectedPatElectrons',
    cut =
    'et > 15. &'
    'abs(eta) < 2.5 &'
    '(dr03TkSumPt+dr03EcalRecHitSumEt+dr03HcalTowerSumEt)/et <  0.2'
 )

from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *
goodJets = selectedPatJets.clone(
    src = 'selectedPatJets',
    cut =
    'pt > 30. &'
    'abs(eta) < 2.4 &'
    'emEnergyFraction > 0.01 &'
    'jetID.n90Hits > 1 &'
    'jetID.fHPD < 0.98'
)

topObjectSelection = cms.Sequence(
    goodJets   *
    vetoElecs  *
    vetoMuons  *
    looseMuons *
    tightMuons
 )
