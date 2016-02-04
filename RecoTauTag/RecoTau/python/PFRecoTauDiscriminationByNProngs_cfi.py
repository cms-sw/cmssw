import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByNProngs = cms.EDProducer("PFRecoTauDiscriminationByNProngs",
    PFTauProducer       = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,
    BooleanOutput       = cms.bool(True),

    nProngs             = cms.uint32(0), # number of prongs required: 0=1||3, 1, 3
)
