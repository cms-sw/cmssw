import FWCore.ParameterSet.Config as cms

pfRecoTauProducer = cms.EDProducer("PFRecoTauProducer",
    LeadTrack_minPt = cms.double(5.0),
    PVProducer = cms.string('offlinePrimaryVertices'), ## ***    

    ECALSignalConeSizeFormula = cms.string('0.15'), ## **       

    TrackerIsolConeMetric = cms.string('DR'), ## * 

    TrackerSignalConeMetric = cms.string('DR'), ## * 

    ECALSignalConeSize_min = cms.double(0.0),
    MatchingConeMetric = cms.string('DR'), ## * 

    TrackerSignalConeSizeFormula = cms.string('0.07'), ## **   

    HCALSignalConeSize_max = cms.double(0.6),
    MatchingConeSizeFormula = cms.string('0.1'), ## **  

    TrackerIsolConeSize_min = cms.double(0.0),
    GammaCand_minPt = cms.double(1.5),
    HCALSignalConeMetric = cms.string('DR'), ## *  

    ChargedHadrCandLeadChargedHadrCand_tksmaxDZ = cms.double(0.2),
    TrackerIsolConeSize_max = cms.double(0.6),
    TrackerSignalConeSize_max = cms.double(0.6),
    MatchingConeSize_min = cms.double(0.0),
    TrackerSignalConeSize_min = cms.double(0.0),
    ECALIsolConeSize_max = cms.double(0.6),
    HCALIsolConeSizeFormula = cms.string('0.50'), ## **        

    Track_IsolAnnulus_minNhits = cms.uint32(8),
    HCALIsolConeSize_max = cms.double(0.6),
    AreaMetric_recoElements_maxabsEta = cms.double(2.5),
    PFTauTagInfoProducer = cms.InputTag("pfRecoTauTagInfoProducer"),
    ECALIsolConeMetric = cms.string('DR'), ## *  

    ECALIsolConeSizeFormula = cms.string('0.50'), ## **         

    UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint = cms.bool(True),
    JetPtMin = cms.double(0.0),
    LeadChargedHadrCand_minPt = cms.double(5.0),
    ECALSignalConeMetric = cms.string('DR'), ## * 

    TrackLeadTrack_maxDZ = cms.double(0.2),
    Track_minPt = cms.double(1.0),
    HCALIsolConeMetric = cms.string('DR'), ## *  

    TrackerIsolConeSizeFormula = cms.string('0.50'), ## **  

    HCALSignalConeSize_min = cms.double(0.0),
    ECALSignalConeSize_max = cms.double(0.6),
    HCALSignalConeSizeFormula = cms.string('0.10'), ## **       

    ChargedHadrCand_IsolAnnulus_minNhits = cms.uint32(8),
    ChargedHadrCand_minPt = cms.double(1.0),
    UseTrackLeadTrackDZconstraint = cms.bool(True),
    smearedPVsigmaY = cms.double(0.0015),
    #string PVProducer                            = "pixelVertices"      # ***  
    smearedPVsigmaX = cms.double(0.0015),
    smearedPVsigmaZ = cms.double(0.005),
    ECALIsolConeSize_min = cms.double(0.0),
    MatchingConeSize_max = cms.double(0.6),
    NeutrHadrCand_minPt = cms.double(1.0),
    HCALIsolConeSize_min = cms.double(0.0)
)


