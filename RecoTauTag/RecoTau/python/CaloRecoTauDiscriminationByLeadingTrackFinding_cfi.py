import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

caloRecoTauDiscriminationByLeadingTrackFinding = cms.EDProducer("CaloRecoTauDiscriminationByLeadingTrackPtCut",

    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),

    Prediscriminants = noPrediscriminants,

    MinPtLeadingTrack = cms.double(0.) # test for existence, not pt
)

