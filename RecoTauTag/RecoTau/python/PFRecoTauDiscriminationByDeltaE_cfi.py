import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByDeltaE = cms.EDProducer("PFRecoTauDiscriminationByDeltaE",
    PFTauProducer       = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    Prediscriminants    = requireLeadTrack,
    BooleanOutput	= cms.bool(True),

    deltaEmin		= cms.double(-0.15),
    deltaEmax           = cms.double(1.0),
)
