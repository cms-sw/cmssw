import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByIsolationUsingLeadingPion = cms.EDProducer("PFRecoTauDiscriminationByIsolationUsingLeadingPion",
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    # following parameters are considered when ManipulateTracks_insteadofChargedHadrCands paremeter is set true
    # *BEGIN*
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    TrackerIsolAnnulus_Candsmaxn = cms.int32(0),
    ECALIsolAnnulus_Candsmaxn = cms.int32(0)
)


