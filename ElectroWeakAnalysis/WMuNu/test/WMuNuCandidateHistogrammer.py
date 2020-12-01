import FWCore.ParameterSet.Config as cms

# Process, how many events, inout files, ...
process = cms.Process("wmunuplots")
process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
       
     # fileNames = cms.untracked.vstring(
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_1.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_2.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_3.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_4.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_5.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_6.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_7.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_8.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_9.root',
     #         '/store/user/cepeda/mytestSkim_PTR_Wmunu_10pb/EWK_WMuNu_SubSkim_31Xv3_10.root'
     #)                  
      
     fileNames = cms.untracked.vstring(
         "file:EWK_WMuNu_SubSkim_31Xv3.root"
      #   "file:AOD_with_WCandidates.root"
     )
)

# Debug/info printouts
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring(
        'corMetWMuNus', 
        'selcorMet'
    )
)

process.selcorMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(True),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT8E29"),
      JetTag = cms.untracked.InputTag("antikt5CaloJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("corMetWMuNus"),

      # Preselection! 
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts ->
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(False),
      IsoCut03 = cms.untracked.double(0.1),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)  
      SelectByCharge=cms.untracked.int32(0)

)
process.selpfMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(True),

      # Preselection! 
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT8E29"),
      JetTag = cms.untracked.InputTag("antikt5CaloJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("pfMetWMuNus"),

      # Main cuts ->
      UseTrackerPt = cms.untracked.bool(True),
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(False),
      IsoCut03 = cms.untracked.double(0.1),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)
      SelectByCharge=cms.untracked.int32(0)

)
process.seltcMet = cms.EDFilter("WMuNuSelector",
      # Fill Basc Histograms? ->
      plotHistograms = cms.untracked.bool(True),

      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      TrigTag = cms.untracked.InputTag("TriggerResults::HLT8E29"),
      JetTag = cms.untracked.InputTag("antikt5CaloJets"),
      WMuNuCollectionTag = cms.untracked.InputTag("tcMetWMuNus"),

      # Preselection! 
      MuonTrig = cms.untracked.string("HLT_Mu9"),
      PtThrForZ1 = cms.untracked.double(20.0),
      PtThrForZ2 = cms.untracked.double(10.0),
      EJetMin = cms.untracked.double(40.),
      NJetMax = cms.untracked.int32(999999),

      # Main cuts ->
      UseTrackerPt = cms.untracked.bool(True),
      PtCut = cms.untracked.double(25.0),
      EtaCut = cms.untracked.double(2.1),
      IsRelativeIso = cms.untracked.bool(True),
      IsCombinedIso = cms.untracked.bool(False),
      IsoCut03 = cms.untracked.double(0.1),
      MtMin = cms.untracked.double(50.0),
      MtMax = cms.untracked.double(200.0),
      MetMin = cms.untracked.double(-999999.),
      MetMax = cms.untracked.double(999999.),
      AcopCut = cms.untracked.double(2.),

      # Muon quality cuts ->
      DxyCut = cms.untracked.double(0.2),
      NormalizedChi2Cut = cms.untracked.double(10.),
      TrackerHitsCut = cms.untracked.int32(11),
      IsAlsoTrackerMuon = cms.untracked.bool(True),

      # Select only W-, W+ ( default is all Ws)
      SelectByCharge=cms.untracked.int32(0)

)

process.TFileService = cms.Service("TFileService", fileName = cms.string('WMuNuBasicPlots.root') )


# Steering the process
process.path1 = cms.Path(process.selcorMet)
process.path2 = cms.Path(process.selpfMet)
process.path3 = cms.Path(process.seltcMet)



