import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

caloRecoTauDiscriminationByFlightPathSignificance = cms.EDProducer("CaloRecoTauDiscriminationByFlightPathSignificance",
    CaloTauProducer     = cms.InputTag('caloRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,
    BooleanOutput       = cms.bool(True),

    flightPathSig	= cms.double(1.5),   # used only if threeProngSelection == true
    UsePVerror		= cms.bool(True),

    qualityCuts         = PFTauQualityCuts,# set the standard quality cuts
    PVProducer          = PFTauQualityCuts.primaryVertexSrc # needed for quality cuts
)
