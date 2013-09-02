import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.fwliteInput.fileNames = cms.vstring(
    'reweightTreeAntiElectronDiscrMVA_signal.root',
    'reweightTreeAntiElectronDiscrMVA_background.root'
)

process.trainTauIdMVA = cms.PSet(

    treeName = cms.string('reweightedAntiElectronDiscrMVATrainingNtuple'),

    signalSamples = cms.vstring('signal'),
    backgroundSamples = cms.vstring('background'),

    applyPtReweighting = cms.bool(True),
    applyEtaReweighting = cms.bool(True),
    reweight = cms.string("flat"),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    applyEventPruning = cms.int32(0),

    mvaName = cms.string("mvaAntiElectronDiscr"),
    mvaMethodType = cms.string("BDT"),
    mvaMethodName = cms.string("BDTG"),
    mvaTrainingOptions = cms.string(
        "!H:!V:NTrees=600:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5"
    ),
    inputVariables = cms.vstring(

    ),
    spectatorVariables = cms.vstring(
        'Tau_Pt/F',
        'Tau_Eta/F',
        'Tau_DecayMode/F',
        'Tau_LeadHadronPt/F',
        'Tau_LooseComb3HitsIso/F',
        'NumPV/I'        
    ),

    outputFileName = cms.string('trainAntiElectronDiscrMVA.root')
)
