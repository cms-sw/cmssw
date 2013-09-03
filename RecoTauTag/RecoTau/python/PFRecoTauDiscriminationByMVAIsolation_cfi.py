import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByMVAIsolation = cms.EDProducer(
    "PFRecoTauDiscriminationByMVAIsolation",
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),
    rhoProducer = cms.InputTag('kt6PFJetsForRhoComputationVoronoi','rho'),
    Prediscriminants = requireLeadTrack,
    gbrfFilePath = cms.FileInPath('RecoTauTag/RecoTau/data/gbrfTauIso.root'),
    returnMVA = cms.bool(True),
    mvaMin = cms.double(0.863),
    )
