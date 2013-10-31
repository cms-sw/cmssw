import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.fwliteInput.fileNames = cms.vstring(
    'preselectTreeAntiElectronDiscrMVA_signal.root',
    'preselectTreeAntiElectronDiscrMVA_background.root'
)

process.reweightTreeTauIdMVA = cms.PSet(

    inputTreeName = cms.string('preselectedAntiElectronDiscrMVATrainingNtuple'),
    outputTreeName = cms.string('reweightedAntiElectronDiscrMVATrainingNtuple'),

    signalSamples = cms.vstring('signal'),
    backgroundSamples = cms.vstring('background'),

    applyPtReweighting = cms.bool(False),
    branchNamePt = cms.string('Tau_Pt'),
    applyEtaReweighting = cms.bool(False),
    branchNameEta = cms.string('Tau_Eta'),
    reweight = cms.string("none"),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    keepAllBranches = cms.bool(True),
    checkBranchesForNaNs = cms.bool(False),

    inputVariables = cms.vstring(),
    spectatorVariables = cms.vstring(
        'Tau_Pt/F',
        'Tau_Eta/F',
        'Tau_DecayMode/F',
        'Tau_LeadHadronPt/F',
        'Tau_LooseComb3HitsIso/F',
        'NumPV/I'
    ),

    outputFileName = cms.string('reweightTreeAntiElectronDiscrMVA.root'),
    save = cms.string('signal')
)
