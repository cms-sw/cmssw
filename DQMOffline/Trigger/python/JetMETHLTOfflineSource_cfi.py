import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSource = cms.EDAnalyzer("JetMETHLTOfflineSource",
 
dirname = cms.untracked.string("HLT/JetMET"),
    DQMStore = cms.untracked.bool(True),                      
    verbose = cms.untracked.bool(False),                        
    plotAll = cms.untracked.bool(True),
    plotAllwrtMu = cms.untracked.bool(False), 
    plotEff = cms.untracked.bool(True),
    nameForEff =  cms.untracked.bool(True),
    nameForMon =  cms.untracked.bool(False), 
    fEMF       = cms.untracked.double(0.01),
    feta       = cms.untracked.double(2.6),
    fHPD       = cms.untracked.double(0.98),
    n90Hits    = cms.untracked.double(1),
    pathnameMuon = cms.untracked.vstring("HLT_L1MuOpen_v2"),                      
    pathnameMB = cms.untracked.vstring("HLT_MinBiasPixel_SingleTrack"), 
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    CaloMETCollectionLabel = cms.InputTag("met"),
#    CaloJetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),
    CaloJetCollectionLabel = cms.InputTag("ak5CaloJets"),
        
    processname = cms.string("HLT"),
    paths = cms.untracked.vstring("HLT_L1MuOpen_v2","HLT_MinBiasPixel_SingleTrack","HLT_Jet15U","HLT_Jet30U","HLT_Jet50U","HLT_Jet70U_v2","HLT_Jet100U_v2","HLT_Jet140U_v1","OpenHLT_Jet180U","HLT_DiJetAve15U","HLT_DiJetAve30U","HLT_DiJetAve50U","HLT_DiJetAve70U_v2","HLT_DiJetAve100U_v1","OpenHLT_DiJetAve140","HLT_L1MET20","HLT_MET45","HLT_MET100"),
    pathPairs = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_Jet15U"),
              denompathname = cms.string("HLT_L1MuOpen_v2"),
             ),          

             cms.PSet(
              pathname = cms.string("HLT_Jet30U"),
              denompathname = cms.string("HLT_Jet15U"),
             ),

             cms.PSet(
              pathname = cms.string("HLT_Jet50U"),
              denompathname = cms.string("HLT_Jet30U"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet70U_v2"),
              denompathname = cms.string("HLT_Jet50U"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet100U_v2"),
              denompathname = cms.string("HLT_Jet70U_v2"),  
             ),
              cms.PSet(
              pathname = cms.string("HLT_Jet140U_v1"),
              denompathname = cms.string("HLT_Jet100U_v2"),
             ),
             cms.PSet(
              pathname = cms.string("OpenHLT_Jet180U"),
              denompathname = cms.string("HLT_Jet140U_v1"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_DiJetAve15U"),
              denompathname = cms.string("HLT_L1MuOpen_v2"),
             ),
             cms.PSet(
              pathname = cms.string("HLT_DiJetAve30U"),
              denompathname = cms.string("HLT_DiJetAve15U"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_DiJetAve50U"),
              denompathname = cms.string("HLT_DiJetAve30U"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_DiJetAve70U_v2"),
              denompathname = cms.string("HLT_DiJetAve50U"),
             ),
             cms.PSet(
              pathname = cms.string("HLT_DiJetAve100U_v1"),
              denompathname = cms.string("HLT_DiJetAve70U_v2"),
             ),   
            cms.PSet(
              pathname = cms.string("OpenHLT_DiJetAve140"),
              denompathname = cms.string("HLT_DiJetAve100U_v1"),
             ),
            cms.PSet(
              pathname = cms.string("HLT_L1MET20"),
              denompathname = cms.string("HLT_L1MuOpen_v2"),
             ),
             cms.PSet(
              pathname = cms.string("HLT_MET45"),
              denompathname = cms.string("HLT_L1MET20"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_MET100"),
              denompathname = cms.string("HLT_MET45"),  
             )

            ),

       JetIDParams  = cms.PSet(
         useRecHits      = cms.bool(True),
         hbheRecHitsColl = cms.InputTag("hbhereco"),
         hoRecHitsColl   = cms.InputTag("horeco"),
         hfRecHitsColl   = cms.InputTag("hfreco"),
         ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
         eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
     )


                                 #-----
                                 
)

