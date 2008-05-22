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
    MatchingConeSize_min = cms.double(0.0),
    GammaCand_minPt = cms.double(1.5), ##Increased from 1.0 to recover efficiency lost by Gamma Conversions    

    #string PVProducer                            = "pixelVertices"      # ***  
    ElectronPreIDProducer = cms.InputTag("elecpreid"),
    ChargedHadrCandLeadChargedHadrCand_tksmaxDZ = cms.double(0.2),
    TrackerIsolConeSize_max = cms.double(0.6),
    TrackerSignalConeSize_max = cms.double(0.6),
    HCALIsolConeMetric = cms.string('DR'), ## *  

    TrackerSignalConeSize_min = cms.double(0.0),
    ECALIsolConeSize_max = cms.double(0.6),
    HCALIsolConeSizeFormula = cms.string('0.50'), ## **        

    AreaMetric_recoElements_maxabsEta = cms.double(2.5),
    HCALIsolConeSize_max = cms.double(0.6),
    Track_IsolAnnulus_minNhits = cms.uint32(8),
    HCALSignalConeMetric = cms.string('DR'), ## *  

    # * possible metrics : "DR", "angle", "area";
    #   if the "area" metric is chosen, AreaMetric_recoElements_maxabsEta parameter is considered, the area of a cone is increased by increasing the angle of the cone;  
    #   functionnality to use a "DR" signal cone and an "area" isolation outer cone is not available;
    # ** may depend on E(energy) and/or PT(transverse momentum) of the initial PFJet, ex. : "3.0/E" or "3.0/ET" 
    #    if XXXConeSizeFormula>XXXConeSize_max then XXXConeSize_max is the considered cone size ; if XXXConeSizeFormula<XXXConeSize_min then XXXConeSize_min is the considered cone size;  
    # *** a PV is needed for computing a leading (charged hadron PFCand) rec. tk signed transverse impact parameter. 
    # For electron rejection variable
    ElecPreIDLeadTkMatch_maxDR = cms.double(0.01),
    PFTauTagInfoProducer = cms.InputTag("pfRecoTauTagInfoProducer"),
    ECALIsolConeMetric = cms.string('DR'), ## *  

    ECALIsolConeSizeFormula = cms.string('0.50'), ## **         

    UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint = cms.bool(True),
    JetPtMin = cms.double(0.0),
    LeadChargedHadrCand_minPt = cms.double(5.0),
    ECALSignalConeMetric = cms.string('DR'), ## * 

    EcalStripSumE_deltaPhiOverQ_maxValue = cms.double(0.5),
    Track_minPt = cms.double(1.0),
    EcalStripSumE_deltaPhiOverQ_minValue = cms.double(-0.1),
    EcalStripSumE_minClusEnergy = cms.double(0.1),
    EcalStripSumE_deltaEta = cms.double(0.03),
    TrackerIsolConeSizeFormula = cms.string('0.50'), ## **  

    HCALSignalConeSize_min = cms.double(0.0),
    ECALSignalConeSize_max = cms.double(0.6),
    HCALSignalConeSizeFormula = cms.string('0.10'), ## **       

    TrackLeadTrack_maxDZ = cms.double(0.2),
    ChargedHadrCand_IsolAnnulus_minNhits = cms.uint32(8),
    ChargedHadrCand_minPt = cms.double(1.0),
    UseTrackLeadTrackDZconstraint = cms.bool(True),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    smearedPVsigmaZ = cms.double(0.005),
    ECALIsolConeSize_min = cms.double(0.0),
    MatchingConeSize_max = cms.double(0.6),
    NeutrHadrCand_minPt = cms.double(1.0),
    HCALIsolConeSize_min = cms.double(0.0)
)


