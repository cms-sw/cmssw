import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.fwliteInput.fileNames = cms.vstring(
    'preselectTreeTauIdMVA_signal.root',
    'preselectTreeTauIdMVA_background.root'
)

process.reweightTreeTauIdMVA = cms.PSet(

    inputTreeName = cms.string('preselectedTauIdMVATrainingNtuple'),
    outputTreeName = cms.string('reweightedTauIdMVATrainingNtuple'),

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

    outputFileName = cms.string('reweightTreeTauIdMVA.root'),
    save = cms.string('signal')
)
