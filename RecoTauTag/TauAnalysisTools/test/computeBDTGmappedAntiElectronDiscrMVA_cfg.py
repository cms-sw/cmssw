import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    ##maxEvents = cms.int32(100000),
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

process.computeBDTGmappedAntiElectronDiscrMVA = cms.PSet(

    inputTreeName = cms.string('extendedTree'),
    outputTreeName = cms.string('extBDTGmappedAntiElectronDiscrMVATrainingNtuple'),

    branchName_mvaOutput = cms.string('BDTG'),
    branchName_categoryIdx = cms.string('Tau_Category'),
    branchName_tauPt = cms.string(''),
    branchName_logTauPt = cms.string('TMath_Log_TMath_Max_1.,Tau_Pt__'),

    categories = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),

    wpFileName = cms.string('computeWPcutsAntiElectronDiscrMVA.root'),
    wpTreeName = cms.string('wpCutsTree'),

    outputFileName = cms.string('computeBDTGmappedAntiElectronDiscrMVA.root')
)
