import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrackCalo

caloRecoTauDiscriminationByLeadingTrackPtCut = cms.EDProducer("CaloRecoTauDiscriminationByLeadingTrackPtCut",

    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),

    Prediscriminants = requireLeadTrackCalo,

    MinPtLeadingTrack = cms.double(5.0)
)


