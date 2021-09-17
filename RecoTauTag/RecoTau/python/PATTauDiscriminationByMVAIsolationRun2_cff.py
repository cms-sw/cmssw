import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminantCutMultiplexer_cfi import *

import RecoTauTag.RecoTau.patTauDiscriminationByMVAIsolationRun2_cfi as _mod
# make sure to load the database containing the mva inputs before using the producers below
# e.g. process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi') as in
# RecoTauTag.Configuration.HPSPFTaus_cff

patDiscriminationByIsolationMVArun2v1raw = _mod.patTauDiscriminationByMVAIsolationRun2.clone(
    # tau collection to discriminate
    PATTauProducer = 'replaceMeByTauCollectionToBeUsed', # in MiniAOD: slimmedTaus
    Prediscriminants = noPrediscriminants,
    loadMVAfromDB = True,
    inputFileName = "RecoTauTag/RecoTau/data/emptyMVAinputFile", # the filename for MVA if it is not loaded from DB
    mvaName = "replaceMeByNameOfMVATraining", # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1
    mvaOpt = "replaceMeByMVAOption", # e.g. DBoldDMwLT
    # change these only if input isolation sums changed for the MVA training you want to use
    srcChargedIsoPtSum = 'chargedIsoPtSum',
    srcNeutralIsoPtSum = 'neutralIsoPtSum',
    srcPUcorrPtSum = 'puCorrPtSum',
    srcPhotonPtSumOutsideSignalCone = 'photonPtSumOutsideSignalCone',
    srcFootprintCorrection = 'footprintCorrection',
)

patDiscriminationByIsolationMVArun2v1 = patTauDiscriminantCutMultiplexer.clone(
    PATTauProducer = 'replaceMeByTauCollectionToBeUsed', # in MiniAOD: slimmedTaus
    Prediscriminants = noPrediscriminants,
    toMultiplex = 'patDiscriminationByIsolationMVArun2v1raw',
    loadMVAfromDB = True,
    mvaOutput_normalization = "replaceMeByNormalizationToBeUsedIfAny", # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("replaceMeByCut"), # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff90
            variable = cms.string("pt"),
        )
    ),
    workingPoints = [
        "Eff80",
        "Eff70",
        "Eff60",
        "Eff50",
        "Eff40"
    ]
)

mvaIsolation2TaskRun2 = cms.Task(
   patDiscriminationByIsolationMVArun2v1raw
   , patDiscriminationByIsolationMVArun2v1
)
mvaIsolation2SeqRun2 = cms.Sequence(mvaIsolation2TaskRun2)
