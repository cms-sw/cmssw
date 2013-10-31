import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    ##maxEvents = cms.int32(100000),
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.preselectTreeTauIdMVA = cms.PSet(

    inputTreeName = cms.string('extendedTree'),
    outputTreeName = cms.string('preselectedAntiElectronDiscrMVATrainingNtuple'),

    preselection = cms.string(''),

    samples = cms.vstring(),
    
    branchNamePt = cms.string('Tau_Pt'),
    branchNameEta = cms.string('Tau_Eta'),
    branchNameNumMatches = cms.string(''),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    applyEventPruning = cms.int32(0),

    keepAllBranches = cms.bool(True),
    checkBranchesForNaNs = cms.bool(False),

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

    outputFileName = cms.string('preselectTreeAntiElectronDiscrMVA.root')
)
