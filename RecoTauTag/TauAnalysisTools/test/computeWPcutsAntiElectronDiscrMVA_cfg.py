import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    ##maxEvents = cms.int32(100000),
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.computeWPcutsAntiElectronDiscrMVA = cms.PSet(

    inputTreeName = cms.string('extendedTree'),
    outputTreeName = cms.string('wpCutsTree'),

    branchName_mvaOutput = cms.string('BDTG'),
    branchName_categoryIdx = cms.string('Tau_Category'),
    branchName_tauPt = cms.string(''),
    branchName_logTauPt = cms.string('TMath_Log_TMath_Max_1.,Tau_Pt__'),
    branchName_evtWeight = cms.string('weight'),
    branchName_classId = cms.string('classID'),
    classId_signal = cms.int32(0),
    classId_background = cms.int32(1),

    categories = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    ptBinning = cms.vdouble(0., 60., 100., 200., 1.e+6),

    outputFileName = cms.string('computeWPcutsAntiElectronDiscrMVA.root')
)
