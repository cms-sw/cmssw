import FWCore.ParameterSet.Config as cms

#from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import *
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

caloRecoTauDiscriminationByDeltaE = cms.EDProducer("CaloRecoTauDiscriminationByDeltaE",
    TauProducer         = cms.InputTag('caloRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,
    BooleanOutput	= cms.bool(True),

    deltaEmin		= cms.double(-0.15),
    deltaEmax           = cms.double(1.0),  
)
