import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByFlightPathSignificance = cms.EDProducer("PFRecoTauDiscriminationByFlightPathSignificance",
    PFTauProducer       = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,
    BooleanOutput       = cms.bool(True),

    flightPathSig	= cms.double(1.5),   # used only if threeProngSelection == true
    UsePVerror		= cms.bool(True),

    PVProducer          = PFTauQualityCuts.primaryVertexSrc,
)
