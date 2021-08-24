import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.recoTauDiscriminantCutMultiplexerDefault_cfi import recoTauDiscriminantCutMultiplexerDefault
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauBasicDiscriminators
import RecoTauTag.RecoTau.pfRecoTauDiscriminationByIsolationMVA2_cfi as _mod

discriminationByIsolationMVA2raw = _mod.pfRecoTauDiscriminationByIsolationMVA2.clone(
    # tau collection to discriminate
    PFTauProducer = 'pfTauProducer',

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,
    loadMVAfromDB = True,
    inputFileName = "RecoTauTag/RecoTau/data/emptyMVAinputFile", # the filename for MVA if it is not loaded from DB
    mvaName = "tauIdMVAnewDMwLT",
    mvaOpt  = "newDMwLT",

    # NOTE: tau lifetime reconstruction sequence needs to be run before
    srcTauTransverseImpactParameters = '',
    
    srcBasicTauDiscriminators = 'hpsPFTauBasicDiscriminators',
    srcChargedIsoPtSumIndex = 0,
    srcNeutralIsoPtSumIndex = 1,
    srcPUcorrPtSumIndex = 5,
)

discriminationByIsolationMVA2 = recoTauDiscriminantCutMultiplexerDefault.clone(
    PFTauProducer = 'pfTauProducer',    
    Prediscriminants = requireLeadTrack,
    toMultiplex = 'discriminationByIsolationMVA2raw',
    loadMVAfromDB = True,
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwLT"),
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

mvaIsolation2Task = cms.Task(
    hpsPFTauBasicDiscriminators
   , discriminationByIsolationMVA2raw
   , discriminationByIsolationMVA2
)
mvaIsolation2Seq = cms.Sequence(mvaIsolation2Task)
