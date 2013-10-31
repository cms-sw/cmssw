import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.fwliteInput.fileNames = cms.vstring(
    'reweightTreeAntiMuonDiscrMVA_signal.root',
    'reweightTreeAntiMuonDiscrMVA_background.root'
)

process.trainTauIdMVA = cms.PSet(

    treeName = cms.string('reweightedAntiMuonDiscrMVATrainingNtuple'),

    signalSamples = cms.vstring('signal'),
    backgroundSamples = cms.vstring('background'),

    applyPtReweighting = cms.bool(True),
    applyEtaReweighting = cms.bool(True),
    reweight = cms.string("flat"),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    applyEventPruning = cms.int32(0),

    mvaName = cms.string("mvaAntiMuonDiscrOpt1a"),
    mvaMethodType = cms.string("BDT"),
    mvaMethodName = cms.string("BDTG"),
    mvaTrainingOptions = cms.string(
        "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5"
    ),
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

    outputFileName = cms.string('trainAntiMuonDiscrMVA.root')
)
