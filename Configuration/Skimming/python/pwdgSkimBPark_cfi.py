import FWCore.ParameterSet.Config as cms


SkimBPark = cms.EDFilter('LeptonSkimming',
     beamSpot = cms.InputTag('offlineBeamSpot'),
     electrons    = cms.InputTag("lowPtGsfElectrons"),
     vertices     = cms.InputTag("offlinePrimaryVertices"),
     eleBiasedWP = cms.InputTag("lowPtGsfElectronSeedValueMaps","ptbiased"),
     eleUnbiasedWP = cms.InputTag("lowPtGsfElectronSeedValueMaps","unbiased"),
     conversions  = cms.InputTag('allConversions'),
     HLTFilter=cms.vstring(
       #V2.0
       "hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8p5Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered10p5Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8Q",
       #V2.2
       "hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered12Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP5Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered7IP4Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP4Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8IP5Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8IP6Q",
       #V3.5
       "hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP3Q","hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP0Q"
),
     HLTPath=cms.vstring(
       #V2.0
       "HLT_Mu9_IP6_part","HLT_Mu8p5_IP3p5","HLT_Mu10p5_IP3p5","HLT_Mu8_IP3",
       #V2.2
       "HLT_Mu12_IP6","HLT_Mu9_IP5","HLT_Mu7_IP4","HLT_Mu9_IP4","HLT_Mu8_IP5","HLT_Mu8_IP6",
       #V3.5
       "HLT_Mu9_IP3","HLT_Mu9_IP0"), 
     triggerresults = cms.InputTag("TriggerResults::HLT"),
     triggerobjects = cms.InputTag('hltTriggerSummaryAOD','','HLT'),            
     muons=cms.InputTag("muons"),
     tracks=cms.InputTag("generalTracks"),                               
     RunParameters = cms.PSet(
       #general
        MuTrgMatchCone=cms.double(0.02),
        #single track related
        PtTrack_Cut=cms.double(0.0),EtaTrack_Cut=cms.double(2.5),TrackMuDz_Cut=cms.double(0.7),
        MinChi2Track_Cut=cms.double(0),MaxChi2Track_Cut=cms.double(1000),
        MuTrkMinDR_Cut=cms.double(0),TrackSdxy_Cut=cms.double(0),
        TrgExclusionCone=cms.double(0.4),
        #object sepecific
        MuTrgExclusionCone=cms.double(0.4),ElTrgExclusionCone=cms.double(0.4),
        TrkObjExclusionCone=cms.double(0.02),PtEl_Cut=cms.double(0),
        PtMu_Cut=cms.double(0.0),QualMu_Cut=cms.double(0),
        MuTrgMuDz_Cut=cms.double(0.3),ElTrgMuDz_Cut=cms.double(0.3),
        SkimOnlyMuons=cms.bool(False), SkimOnlyElectrons=cms.bool(True),
        BiasedWP=cms.double(-100), UnbiasedWP=cms.double(6.0),
        #trkobj pair related
        MaxMee_Cut=cms.double(500),EpairZvtx_Cut=cms.double(1000),
        MinMee_Cut=cms.double(0),Cosee_Cut=cms.double(-10.99),
        Probee_Cut=cms.double(-1),        
        ObjPtLargerThanTrack=cms.bool(False),
        #triple obj-track
        PtKTrack_Cut=cms.double(-10), Ksdxy_Cut=cms.double(0),
        MaxMB_Cut=cms.double(6000.0),MinMB_Cut=cms.double(-10),
        SLxy_Cut=cms.double(-10),ProbeeK_Cut=cms.double(-10),
        CoseeK_Cut=cms.double(-10), PtB_Cut=cms.double(-10),
        #configuration
        SkipIfNoMuMatch=cms.bool(False)
     )
)

