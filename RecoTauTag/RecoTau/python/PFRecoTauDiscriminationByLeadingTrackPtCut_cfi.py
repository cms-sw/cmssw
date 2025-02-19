import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

pfRecoTauDiscriminationByLeadingTrackPtCut = cms.EDProducer("PFRecoTauDiscriminationByLeadingObjectPtCut",

    # Tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # no pre-reqs for this cut
    Prediscriminants = noPrediscriminants,

    # Allow only charged PFCandidates to meet this requirement
    UseOnlyChargedHadrons = cms.bool(True),            

    MinPtLeadingObject = cms.double(5.0)
)


