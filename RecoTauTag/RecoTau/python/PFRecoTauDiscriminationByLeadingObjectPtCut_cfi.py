import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

pfRecoTauDiscriminationByLeadingObjectPtCut = cms.EDProducer("PFRecoTauDiscriminationByLeadingObjectPtCut",

    # Tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # no pre-reqs for this cut
    Prediscriminants = noPrediscriminants,

    # Allow either charged or neutral PFCandidates to meet this requirement
    UseOnlyChargedHadrons = cms.bool(False),            

    MinPtLeadingObject = cms.double(5.0)
)


