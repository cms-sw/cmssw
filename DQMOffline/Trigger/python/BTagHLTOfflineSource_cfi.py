import FWCore.ParameterSet.Config as cms

btagHLTOfflineSource = cms.EDAnalyzer("BTagHLTOfflineSource",
 
dirname = cms.untracked.string("HLT/BTagMu"),
    DQMStore = cms.untracked.bool(True),                      
    verbose = cms.untracked.bool(False),

    plotEff = cms.untracked.bool(True),
    nameForEff =  cms.untracked.bool(True),
    nameForMon =  cms.untracked.bool(False), 

    jetpt      = cms.untracked.double(20.),
    jeteta     = cms.untracked.double(2.4),
    fEMF       = cms.untracked.double(0.01),
    fHPD       = cms.untracked.double(0.98),
    n90Hits    = cms.untracked.double(1),

    mupt              = cms.untracked.double(6.),
    mueta             = cms.untracked.double(2.4),
    muonHits          = cms.untracked.int32(0),
    nMatches          = cms.untracked.int32(1),
    trackerHits       = cms.untracked.int32(10),
    pixelHits         = cms.untracked.int32(1),
    outerHits         = cms.untracked.int32(3),
    tknormalizedChi2  = cms.untracked.double(10),
    gmnormalizedChi2  = cms.untracked.double(10),
    mudZ              = cms.untracked.double(2),
    mujetdR           = cms.untracked.double(0.4),

    pathnameMuon = cms.untracked.vstring("HLT_Jet15U_v3"),                      
    pathnameMB = cms.untracked.vstring("HLT_MinBiasBSC"), 
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    CaloJetCollectionLabel = cms.InputTag("ak5CaloJets"),
    MuonCollectionLabel    = cms.InputTag("muons"),
        
    processname = cms.string("HLT"),
    paths = cms.untracked.vstring("HLT_Jet15U_v3","HLT_MinBiasBSC","HLT_BTagMu_DiJet10U_v3","HLT_BTagMu_DiJet20U_v3", "HLT_BTagMu_DiJet20U_Mu5_v3", "HLT_BTagMu_DiJet30U_v3", "HLT_BTagMu_DiJet30U_Mu5_v3"),
    pathPairs = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_BTagMu_DiJet10U_v3"),
              denompathname = cms.string("HLT_Jet15U_v3"),
             ),
             cms.PSet(
              pathname = cms.string("HLT_BTagMu_DiJet20U_v3"),
              denompathname = cms.string("HLT_BTagMu_DiJet10U_v3"),
             ),
             cms.PSet(
              pathname = cms.string("HLT_BTagMu_DiJet30U_v3"),
              denompathname = cms.string("HLT_BTagMu_DiJet20U_v3"),
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
                                 
)

