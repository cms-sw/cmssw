import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingObjectPtCut_cfi import pfRecoTauDiscriminationByLeadingObjectPtCut

pfRecoTauDiscriminationByLeadingTrackFinding = pfRecoTauDiscriminationByLeadingObjectPtCut.clone(
    # Tau collection to discriminate
    PFTauProducer = 'pfRecoTauProducer',

    # Only look for charged PFCandidates
    UseOnlyChargedHadrons = True,            

    # no pre-reqs for this cut
    Prediscriminants = noPrediscriminants,

    # Any *existing* charged PFCandidate will meet this requirement - so it is 
    # a test for existence, not pt
    MinPtLeadingObject = 0.0
)

