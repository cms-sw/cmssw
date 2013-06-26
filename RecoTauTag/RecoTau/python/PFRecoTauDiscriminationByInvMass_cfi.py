import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByInvMass = cms.EDProducer("PFRecoTauDiscriminationByInvMass",
    PFTauProducer       = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,

    select = cms.PSet(
        min = cms.double(0.0),
        max = cms.double(1.4),
    )
)
