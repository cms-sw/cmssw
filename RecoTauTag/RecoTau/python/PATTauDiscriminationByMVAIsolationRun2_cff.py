import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminantCutMultiplexer_cfi import *

# make sure to load the database containing the mva inputs before using the producers below
# e.g. process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi') as in
# RecoTauTag.Configuration.HPSPFTaus_cff

patDiscriminationByIsolationMVArun2v1raw = cms.EDProducer("PATTauDiscriminationByMVAIsolationRun2",

    # tau collection to discriminate
    PATTauProducer = cms.InputTag('replaceMeByTauCollectionToBeUsed'), # in MiniAOD: slimmedTaus
    Prediscriminants = noPrediscriminants,
    loadMVAfromDB = cms.bool(True),
    inputFileName = cms.FileInPath("RecoTauTag/RecoTau/data/emptyMVAinputFile"), # the filename for MVA if it is not loaded from DB
    mvaName = cms.string("replaceMeByNameOfMVATraining"), # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1
    mvaOpt = cms.string("replaceMeByMVAOption"), # e.g. DBoldDMwLT
    
    # change these only if input isolation sums changed for the MVA training you want to use
    srcChargedIsoPtSum = cms.string('chargedIsoPtSum'),
    srcNeutralIsoPtSum = cms.string('neutralIsoPtSum'),
    srcPUcorrPtSum = cms.string('puCorrPtSum'),
    srcPhotonPtSumOutsideSignalCone = cms.string('photonPtSumOutsideSignalCone'),
    srcFootprintCorrection = cms.string('footprintCorrection'),

    verbosity = cms.int32(0)
)

patDiscriminationByIsolationMVArun2v1VLoose = patTauDiscriminantCutMultiplexer.clone(
    PATTauProducer = cms.InputTag('replaceMeByTauCollectionToBeUsed'), # in MiniAOD: slimmedTaus
    Prediscriminants = noPrediscriminants,
    toMultiplex = cms.InputTag('patDiscriminationByIsolationMVArun2v1raw'),
    key = cms.InputTag('patDiscriminationByIsolationMVArun2v1raw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("replaceMeByNormalizationToBeUsedIfAny"), # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("replaceMeByCut"), # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff90
            variable = cms.string("pt"),
        )
    )
)
patDiscriminationByIsolationMVArun2v1Loose = patDiscriminationByIsolationMVArun2v1VLoose.clone()
patDiscriminationByIsolationMVArun2v1Loose.mapping[0].cut = cms.string("replaceMeByCut") # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff80
patDiscriminationByIsolationMVArun2v1Medium = patDiscriminationByIsolationMVArun2v1VLoose.clone()
patDiscriminationByIsolationMVArun2v1Medium.mapping[0].cut = cms.string("replaceMeByCut") # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff70
patDiscriminationByIsolationMVArun2v1Tight = patDiscriminationByIsolationMVArun2v1VLoose.clone()
patDiscriminationByIsolationMVArun2v1Tight.mapping[0].cut = cms.string("replaceMeByCut") # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff60
patDiscriminationByIsolationMVArun2v1VTight = patDiscriminationByIsolationMVArun2v1VLoose.clone()
patDiscriminationByIsolationMVArun2v1VTight.mapping[0].cut = cms.string("replaceMeByCut") # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff50
patDiscriminationByIsolationMVArun2v1VVTight = patDiscriminationByIsolationMVArun2v1VLoose.clone()
patDiscriminationByIsolationMVArun2v1VVTight.mapping[0].cut = cms.string("replaceMeByCut") # e.g. RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff40

mvaIsolation2TaskRun2 = cms.Task(
   patDiscriminationByIsolationMVArun2v1raw
   , patDiscriminationByIsolationMVArun2v1VLoose
   , patDiscriminationByIsolationMVArun2v1Loose
   , patDiscriminationByIsolationMVArun2v1Medium
   , patDiscriminationByIsolationMVArun2v1Tight
   , patDiscriminationByIsolationMVArun2v1VTight
   , patDiscriminationByIsolationMVArun2v1VVTight
)
mvaIsolation2SeqRun2 = cms.Sequence(mvaIsolation2TaskRun2)
