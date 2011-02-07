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
    pathnameMB = cms.untracked.vstring("HLT_MinBiasBSC"), 
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    CaloMETCollectionLabel = cms.InputTag("met"),
    CaloJetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),
        
    processname = cms.string("HLT"),
    paths = cms.untracked.vstring("HLT_L1MuOpen_v2","HLT_MinBiasBSC","HLT_L1Jet6U","HLT_L1Jet10U","HLT_Jet15U","HLT_Jet30U","HLT_Jet50U","HLT_DiJetAve15U_8E29","HLT_DiJetAve30U_8E29","HLT_L1MET20","HLT_MET45","HLT_MET100","HLT_HT100U"),
    pathPairs = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_L1Jet6U"),
              denompathname = cms.string("HLT_L1MuOpen_v2"),
             ),          

             cms.PSet(
              pathname = cms.string("HLT_L1Jet10U"),
              denompathname = cms.string("HLT_L1Jet6U"),
             ),

             cms.PSet(
              pathname = cms.string("HLT_Jet15U"),
              denompathname = cms.string("HLT_L1Jet6U"),  
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
              pathname = cms.string("HLT_DiJetAve15U_8E29"),
              denompathname = cms.string("HLT_L1MuOpen_v2"),
             ),
             cms.PSet(
              pathname = cms.string("HLT_DiJetAve30U_8E29"),
              denompathname = cms.string("HLT_DiJetAve15U_8E29"),  
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
             ),
             cms.PSet(
              pathname = cms.string("HLT_HT100U"),
              denompathname = cms.string("HLT_L1MuOpen_v2"),
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

