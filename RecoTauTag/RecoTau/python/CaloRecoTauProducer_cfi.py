import FWCore.ParameterSet.Config as cms

caloRecoTauProducer = cms.EDProducer(
    "CaloRecoTauProducer",
    LeadTrack_minPt = cms.double(1.0),
    #string PVProducer                         = "pixelVertices"      # ***  
    PVProducer = cms.InputTag('offlinePrimaryVertices'),
    ECALSignalConeSizeFormula = cms.string('0.15'), ## **       

    TrackerIsolConeMetric = cms.string('DR'), ## *  

    TrackerSignalConeMetric = cms.string('DR'), ## *  

    ECALSignalConeSize_min = cms.double(0.0),
    ECALRecHit_minEt = cms.double(0.5),
    MatchingConeMetric = cms.string('DR'), ## *  

    TrackerSignalConeSizeFormula = cms.string('0.07'), ## **   

    MatchingConeSizeFormula = cms.string('0.10'), ## **   

    TrackerIsolConeSize_min = cms.double(0.0),
    TrackerIsolConeSize_max = cms.double(0.6),
    TrackerSignalConeSize_max = cms.double(0.6),
    MatchingConeSize_min = cms.double(0.0),
    TrackerSignalConeSize_min = cms.double(0.0),
    ECALIsolConeSize_max = cms.double(0.6),
    AreaMetric_recoElements_maxabsEta = cms.double(2.5),
    ECALIsolConeMetric = cms.string('DR'), ## *  

    ECALIsolConeSizeFormula = cms.string('0.50'), ## **         

    JetPtMin = cms.double(0.0),
    ECALSignalConeMetric = cms.string('DR'), ## * 

    TrackLeadTrack_maxDZ = cms.double(0.2),
    Track_minPt = cms.double(0.5),
    TrackerIsolConeSizeFormula = cms.string('0.50'), ## **
    IsolationTrack_minPt = cms.double(1.0),
    IsolationTrack_minHits = cms.uint32(0),

    ECALSignalConeSize_max = cms.double(0.6),
    ECALIsolConeSize_min = cms.double(0.0),
    UseTrackLeadTrackDZconstraint = cms.bool(True),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    smearedPVsigmaZ = cms.double(0.005),
    CaloRecoTauTagInfoProducer = cms.InputTag("caloRecoTauTagInfoProducer"),
    # parameters of the considered EcalRecHits
    ESRecHitsSource = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    EBRecHitsSource = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EERecHitsSource = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    MatchingConeSize_max = cms.double(0.6)
)


