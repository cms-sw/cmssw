import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.fwliteInput.fileNames = cms.vstring(
    'preselectTreeAntiMuonDiscrMVA_signal.root',
    'preselectTreeAntiMuonDiscrMVA_background.root'
)

process.reweightTreeTauIdMVA = cms.PSet(

    inputTreeName = cms.string('preselectedAntiMuonDiscrMVATrainingNtuple'),
    outputTreeName = cms.string('reweightedAntiMuonDiscrMVATrainingNtuple'),

    signalSamples = cms.vstring('signal'),
    backgroundSamples = cms.vstring('background'),

    applyPtReweighting = cms.bool(True),
    branchNamePt = cms.string('recTauPt'),
    applyEtaReweighting = cms.bool(True),
    branchNameEta = cms.string('recTauEta'),
    reweight = cms.string("flat"),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    keepAllBranches = cms.bool(False),
    checkBranchesForNaNs = cms.bool(True),

    inputVariables = cms.vstring(
        'TMath::Abs(TMath::ACos(TMath::Cos(12.*recTauPhi))/12.)/F',
        'TMath::Abs(recTauEta)/F',
        'TMath::Sqrt(TMath::Max(0., leadPFChargedHadrCandCaloEnECAL))/F',
        'TMath::Sqrt(TMath::Max(0., leadPFChargedHadrCandCaloEnHCAL))/F',
        'numMatches/F',
        'numHitsDT1 + numHitsCSC1 + numHitsRPC1/F',
        'numHitsDT2 + numHitsCSC2 + numHitsRPC2/F',
        'numHitsDT3 + numHitsCSC3 + numHitsRPC3/F',
        'numHitsDT4 + numHitsCSC4 + numHitsRPC4/F'
    ),
    spectatorVariables = cms.vstring(
        'recTauPt/F',
        'numOfflinePrimaryVertices/I'
    ),

    outputFileName = cms.string('reweightTreeAntiMuonDiscrMVA.root'),
    save = cms.string('signal')
)
