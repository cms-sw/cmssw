import FWCore.ParameterSet.Config as cms
hltMonjmDQM = cms.EDAnalyzer("HLTJetMETDQMSource",
    dirname = cms.untracked.string("HLT/JetMET"),
    DQMStore = cms.untracked.bool(True),                      
    #MonitorDaemon = cms.untracked.bool(True),
    #DaqMonitorBEInterface = cms.untracked.bool(True), 
    #verbose = cms.untracked.bool(True),                        
    plotAll = cms.untracked.bool(True),
    plotwrtMu = cms.untracked.bool(True),
    plotEff = cms.untracked.bool(True),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    pathnameMuon = cms.untracked.string("HLT_L1MuOpen"),                       
    paths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_L1Mu"),
              denompathname = cms.string("HLT_L1MuOpen"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet15U"),
              denompathname = cms.string("HLT_L1Jet6U"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet30U"),
              denompathname = cms.string("HLT_Jet15U"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet50U"),
              denompathname = cms.string("HLT_Jet30U"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet110U"),
              denompathname = cms.string("HLT_Jet50U"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_DijetAve30U"),
              denompathname = cms.string("HLT_DijetAve15U"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_MET35"),
              denompathname = cms.string("HLT_L1MET20"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_MET45"),
              denompathname = cms.string("HLT_MET35"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_MET100"),
              denompathname = cms.string("HLT_MET45"),  
             prescaleused = cms.int32(10),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_L1Jet6U"),
              denompathname = cms.string("HLT_L1MuOpen"),
             prescaleused = cms.int32(10),  
             ),
            ),
      
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    processname = cms.string("HLT")

)
