import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByTrackIsolation = cms.EDFilter("PFRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByECALIsolation         = cms.bool(False),
    maxGammaPt                                 = cms.double(1.50),
    PFTauProducer                              = cms.InputTag('pfRecoTauProducer'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    TrackerIsolAnnulus_Tracksmaxn              = cms.int32(0),
    ApplyDiscriminationByTrackerIsolation      = cms.bool(True),
    maxChargedPt                               = cms.double(1.0),
    TrackerIsolAnnulus_Candsmaxn               = cms.int32(0),
    ECALIsolAnnulus_Candsmaxn                  = cms.int32(0),
    TrackIsolationOverTauPt                    = cms.bool(False),
    SumOverCandidates                          = cms.bool(False)                                              
)


