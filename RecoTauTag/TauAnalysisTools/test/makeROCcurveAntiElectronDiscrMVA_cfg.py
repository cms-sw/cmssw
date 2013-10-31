import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(1000)
)

process.makeROCcurveTauIdMVA = cms.PSet(

    treeName = cms.string('antiElectronDiscrMVATrainingNtupleProducer/antiElectronDiscrMVATrainingNtuple'),

    preselection = cms.string(''),

    signalSamples = cms.vstring(),
    backgroundSamples = cms.vstring(),

    ##classId_signal = cms.int(0),
    ##classId_background = cms.int(1),
    ##branchNameClassId = cms.string("classID"),

    discriminator = cms.string(''),

    branchNameLogTauPt = cms.string(''),
    branchNameTauPt = cms.string('Tau_Pt'),

    branchNameEvtWeight = cms.string('evtWeight'),

    graphName = cms.string("antiElectronDiscrLoose"),
    binning = cms.PSet(
        numBins = cms.int32(2),
        min = cms.double(-0.5),
        max = cms.double(+1.5)
    ),

    outputFileName = cms.string('makeROCcurveAntiElectronDiscrMVA.root')
)
