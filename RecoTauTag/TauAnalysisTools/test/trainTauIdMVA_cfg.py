import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.fwliteInput.fileNames = cms.vstring(
    'reweightTreeTauIdMVA_signal.root',
    'reweightTreeTauIdMVA_background.root'
)

process.trainTauIdMVA = cms.PSet(

    treeName = cms.string('reweightedTauIdMVATrainingNtuple'),

    signalSamples = cms.vstring('signal'),
    backgroundSamples = cms.vstring('background'),

    applyPtReweighting = cms.bool(True),
    applyEtaReweighting = cms.bool(True),
    reweight = cms.string("flat"),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    applyEventPruning = cms.int32(0),

    mvaName = cms.string("mvaIsolation3HitsDeltaR05opt1"),
    mvaMethodType = cms.string("BDT"),
    mvaMethodName = cms.string("BDTG"),
    mvaTrainingOptions = cms.string(
        "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5"
    ),
    inputVariables = cms.vstring(
        ##'TMath::Log(TMath::Max(1., recTauPt))/F',
        'TMath::Abs(recTauEta)/F',
        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR08PtThresholdsLoose3HitsPUcorrPtSum))/F',
        'recTauDecayMode/I'
    ),
    spectatorVariables = cms.vstring(
        'recTauPt/F',
        ##'recTauDecayMode/I',
        'leadPFChargedHadrCandPt/F',
        'numOfflinePrimaryVertices/I'
    ),

    outputFileName = cms.string('trainTauIdMVA.root')
)
