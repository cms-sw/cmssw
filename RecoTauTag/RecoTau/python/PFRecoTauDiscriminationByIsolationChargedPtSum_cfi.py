import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByIsolationChargedSumPt = cms.EDProducer("PFRecoTauDiscriminationByIsolationChargedSumPt",
    PFTauProducer                              = cms.InputTag('pfRecoTauProducer'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    MaxChargedSumPt                            = cms.double(8.0), 
    MinPtForObjectInclusion                    = cms.double(0.5)
)


