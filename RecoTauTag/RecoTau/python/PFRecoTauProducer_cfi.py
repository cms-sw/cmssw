import FWCore.ParameterSet.Config as cms

pfRecoTauProducer = cms.EDProducer("PFRecoTauProducer",
                                   PFTauTagInfoProducer = cms.InputTag("pfRecoTauTagInfoProducer"),
                                   PVProducer = cms.InputTag('offlinePrimaryVertices'), ## ***    
                                   JetPtMin = cms.double(0.0),

                                   #Parameters for leading track/charged/neutral pion finding
                                   LeadTrack_minPt = cms.double(5.0),
                                   PFCand_minPt = cms.double(0.5),
                                   LeadPFCand_minPt = cms.double(5.0),
                                   ChargedHadrCandLeadChargedHadrCand_tksmaxDZ = cms.double(0.2),
                                   UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint = cms.bool(True),
                                   TrackLeadTrack_maxDZ = cms.double(0.2),
                                   UseTrackLeadTrackDZconstraint = cms.bool(True),
                                   smearedPVsigmaY = cms.double(0.0015),
                                   smearedPVsigmaX = cms.double(0.0015),
                                   smearedPVsigmaZ = cms.double(0.005),

                                   #MatchingCone paramters
                                   MatchingConeMetric = cms.string('DR'), ## *
                                   MatchingConeSizeFormula = cms.string('0.1'), ## **
                                   MatchingConeSize_min = cms.double(0.0),
                                   MatchingConeSize_max = cms.double(0.6),
                                   
                                   #SignalCone parameters
                                   TrackerSignalConeMetric = cms.string('DR'), ## * 
                                   TrackerSignalConeSizeFormula = cms.string('0.07'), ## **   
                                   TrackerSignalConeSize_min = cms.double(0.0),
                                   TrackerSignalConeSize_max = cms.double(0.6),

                                   ECALSignalConeMetric = cms.string('DR'), ## * 
                                   ECALSignalConeSizeFormula = cms.string('0.15'), ## **
                                   ECALSignalConeSize_min = cms.double(0.0),
                                   ECALSignalConeSize_max = cms.double(0.6),
                                   
                                   HCALSignalConeMetric = cms.string('DR'), ## *  
                                   HCALSignalConeSizeFormula = cms.string('0.10'), ## **
                                   HCALSignalConeSize_min = cms.double(0.0),
                                   HCALSignalConeSize_max = cms.double(0.6),
                                   
                                   #IsolationCone parameters
                                   TrackerIsolConeMetric = cms.string('DR'), ## * 
                                   TrackerIsolConeSizeFormula = cms.string('0.50'), ## ** 
                                   TrackerIsolConeSize_min = cms.double(0.0),
                                   TrackerIsolConeSize_max = cms.double(0.6),

                                   ECALIsolConeMetric = cms.string('DR'), ## *  
                                   ECALIsolConeSizeFormula = cms.string('0.50'), ## **         
                                   ECALIsolConeSize_min = cms.double(0.0),
                                   ECALIsolConeSize_max = cms.double(0.6),
    
                                   HCALIsolConeMetric = cms.string('DR'), ## * 
                                   HCALIsolConeSizeFormula = cms.string('0.50'), ## **
                                   HCALIsolConeSize_min = cms.double(0.0),
                                   HCALIsolConeSize_max = cms.double(0.6),

                                   #FixedArea parameters
                                   AreaMetric_recoElements_maxabsEta = cms.double(2.5),

                                   #Selection on isolation candidates
                                   Track_minPt = cms.double(1.0),
                                   Track_IsolAnnulus_minNhits = cms.uint32(3),

                                   ChargedHadrCand_minPt = cms.double(1.0),
                                   ChargedHadrCand_IsolAnnulus_minNhits = cms.uint32(8),

                                   GammaCand_minPt = cms.double(1.5), ##Increased from 1.0 to recover efficiency lost by Gamma Conversions    
                                   NeutrHadrCand_minPt = cms.double(1.0),

                                   #Electron rejection parameters
                                   ElectronPreIDProducer = cms.InputTag("elecpreid"),
                                   EcalStripSumE_deltaPhiOverQ_minValue = cms.double(-0.1),
                                   EcalStripSumE_deltaPhiOverQ_maxValue = cms.double(0.5),
                                   EcalStripSumE_minClusEnergy = cms.double(0.1),
                                   EcalStripSumE_deltaEta = cms.double(0.03),
                                   ElecPreIDLeadTkMatch_maxDR = cms.double(0.01),

                                   #DataType
                                   DataType = cms.string("AOD")


)
 # * possible metrics : "DR", "angle", "area";
    #   if the "area" metric is chosen, AreaMetric_recoElements_maxabsEta parameter is considered, the area of a cone is increased by increasing the angle of the cone;  
    #   functionnality to use a "DR" signal cone and an "area" isolation outer cone is not available;
    # ** may depend on E(energy) and/or PT(transverse momentum) of the initial PFJet, ex. : "3.0/E" or "3.0/ET" 
    #    if XXXConeSizeFormula>XXXConeSize_max then XXXConeSize_max is the considered cone size ; if XXXConeSizeFormula<XXXConeSize_min then XXXConeSize_min is the considered cone size;  
    # *** a PV is needed for computing a leading (charged hadron PFCand) rec. tk signed transverse impact parameter. 
    # For electron rejection variable

