import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import *
from RecoTauTag.RecoTau.pfRecoTauDiscriminationByMVAIsolationRun2_cfi import pfRecoTauDiscriminationByMVAIsolationRun2
from RecoTauTag.RecoTau.recoTauDiscriminantCutMultiplexerDefault_cfi import recoTauDiscriminantCutMultiplexerDefault
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauBasicDiscriminators

discriminationByIsolationMVArun2v1raw = pfRecoTauDiscriminationByMVAIsolationRun2.clone(

    # tau collection to discriminate
    PFTauProducer = 'pfTauProducer',

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,
    loadMVAfromDB = True,
        
    srcBasicTauDiscriminators = 'hpsPFTauBasicDiscriminators'
)

discriminationByIsolationMVArun2v1 = recoTauDiscriminantCutMultiplexerDefault.clone(
    PFTauProducer = 'pfTauProducer',    
    Prediscriminants = requireLeadTrack,
    toMultiplex = 'discriminationByIsolationMVArun2v1raw',
    loadMVAfromDB = True,
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwLT"),
            variable = cms.string("pt"),
        )
    ),
    workingPoints = cms.vstring(
        "Eff80",
        "Eff70",
        "Eff60",
        "Eff50",
        "Eff40"
    )
)

mvaIsolation2TaskRun2 = cms.Task(
    hpsPFTauBasicDiscriminators
   , discriminationByIsolationMVArun2v1raw
   , discriminationByIsolationMVArun2v1
)
mvaIsolation2SeqRun2 = cms.Sequence(mvaIsolation2TaskRun2)
