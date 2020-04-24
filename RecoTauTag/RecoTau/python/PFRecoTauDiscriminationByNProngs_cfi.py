import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import *
#from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts

pfRecoTauDiscriminationByNProngs = cms.EDProducer("PFRecoTauDiscriminationByNProngs",
    PFTauProducer       = cms.InputTag('combinatoricRecoTaus'),

#    Prediscriminants    = requireLeadTrack,
    Prediscriminants    = noPrediscriminants,
    BooleanOutput       = cms.bool(True),

    MaxN                = cms.uint32(0), # number of prongs required: 0=1||3
    MinN		= cms.uint32(1),

    qualityCuts         = PFTauQualityCuts
)
