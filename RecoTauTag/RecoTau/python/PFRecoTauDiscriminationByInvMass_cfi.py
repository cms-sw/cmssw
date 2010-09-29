import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByInvMass = cms.EDProducer("PFRecoTauDiscriminationByInvMass",
    PFTauProducer       = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,
    BooleanOutput	= cms.bool(True),

    invMassMin		= cms.double(0.0),   # used only if threeProngSelection == true
    invMassMax          = cms.double(1.4),   # used only if threeProngSelection == true
)
