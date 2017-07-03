import FWCore.ParameterSet.Config as cms

triggerFlagPSet = cms.PSet(
  dcsInputTag = cms.InputTag('scalersRawToDigi'),
  dcsPartitions = cms.vint32( 24, 25, 26, 27, 28, 29 ),
  andOrDcs = cms.bool(False),
  errorReplyDcs = cms.bool(True),
  dbLabel = cms.string(''),
  andOrHlt = cms.bool(True),
  hltInputTag = cms.InputTag('TriggerResults', '', 'HLT'),
  hltPaths = cms.vstring(),
  hltDBKey = cms.string(''),
  errorReplyHlt = cms.bool(False),
  verbosityLevel = cms.uint32(1)
)

mssmHbbBtagTriggerMonitor = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/"),
    processname = cms.string("HLT"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (12),
          xmin  = cms.double(40),
          xmax  = cms.double(100),
       ),
       jetEta = cms.PSet (
          nbins = cms.int32 (10),
          xmin  = cms.double(-2.5),
          xmax  = cms.double(2.5),
       ),
       jetPhi = cms.PSet (
          nbins = cms.int32 (14),
          xmin  = cms.double(-3.5),
          xmax  = cms.double(3.5),
       ),
       jetBtag = cms.PSet (
          nbins = cms.int32 (20),
          xmin  = cms.double(0.8),
          xmax  = cms.double(1),
       ),
    ),
    genericTriggerEventPSet = triggerFlagPSet.clone(),
)


# online btagging monitor

mssmHbbBtagTriggerMonitorSL40noMu = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL40noMu.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40_noMuon")
mssmHbbBtagTriggerMonitorSL40noMu.jetPtMin = cms.double(40)
mssmHbbBtagTriggerMonitorSL40noMu.triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorSL40noMu.histoPSet.jetPt.nbins = cms.int32 (12)
mssmHbbBtagTriggerMonitorSL40noMu.histoPSet.jetPt.xmin  = cms.double(40)
mssmHbbBtagTriggerMonitorSL40noMu.histoPSet.jetPt.xmax  = cms.double(100)
mssmHbbBtagTriggerMonitorSL40noMu.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_SingleBTagCSV_0p92_DoublePFJets40_v*')

mssmHbbBtagTriggerMonitorSL40 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL40.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40")
mssmHbbBtagTriggerMonitorSL40.jetPtMin = cms.double(40)
mssmHbbBtagTriggerMonitorSL40.triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorSL40.histoPSet.jetPt.nbins = cms.int32 (12)
mssmHbbBtagTriggerMonitorSL40.histoPSet.jetPt.xmin  = cms.double(40)
mssmHbbBtagTriggerMonitorSL40.histoPSet.jetPt.xmax  = cms.double(100)
mssmHbbBtagTriggerMonitorSL40.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets40_v*')

mssmHbbBtagTriggerMonitorSL100 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL100.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt100")
mssmHbbBtagTriggerMonitorSL100.jetPtMin = cms.double(100)
mssmHbbBtagTriggerMonitorSL100.triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorSL100.histoPSet.jetPt.nbins = cms.int32 (10)
mssmHbbBtagTriggerMonitorSL100.histoPSet.jetPt.xmin  = cms.double(100)
mssmHbbBtagTriggerMonitorSL100.histoPSet.jetPt.xmax  = cms.double(200)
mssmHbbBtagTriggerMonitorSL100.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets100_v*')

mssmHbbBtagTriggerMonitorSL200 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL200.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt200")
mssmHbbBtagTriggerMonitorSL200.jetPtMin = cms.double(200)
mssmHbbBtagTriggerMonitorSL200.triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorSL200.histoPSet.jetPt.nbins = cms.int32 (15)
mssmHbbBtagTriggerMonitorSL200.histoPSet.jetPt.xmin  = cms.double(200)
mssmHbbBtagTriggerMonitorSL200.histoPSet.jetPt.xmax  = cms.double(350)
mssmHbbBtagTriggerMonitorSL200.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets200_v*')

mssmHbbBtagTriggerMonitorSL350 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorSL350.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt350")
mssmHbbBtagTriggerMonitorSL350.jetPtMin = cms.double(350)
mssmHbbBtagTriggerMonitorSL350.triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorSL350.histoPSet.jetPt.nbins = cms.int32 (25)
mssmHbbBtagTriggerMonitorSL350.histoPSet.jetPt.xmin  = cms.double(350)
mssmHbbBtagTriggerMonitorSL350.histoPSet.jetPt.xmax  = cms.double(600)
mssmHbbBtagTriggerMonitorSL350.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets200_v*')

mssmHbbBtagTriggerMonitorAH100 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorAH100.dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt100")
mssmHbbBtagTriggerMonitorAH100.jetPtMin = cms.double(100)
mssmHbbBtagTriggerMonitorAH100.triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorAH100.histoPSet.jetPt.nbins = cms.int32 (10)
mssmHbbBtagTriggerMonitorAH100.histoPSet.jetPt.xmin  = cms.double(100)
mssmHbbBtagTriggerMonitorAH100.histoPSet.jetPt.xmax  = cms.double(200)
mssmHbbBtagTriggerMonitorAH100.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets100_v*')

mssmHbbBtagTriggerMonitorAH200 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorAH200.dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt200")
mssmHbbBtagTriggerMonitorAH200.jetPtMin = cms.double(200)
mssmHbbBtagTriggerMonitorAH200.triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorAH200.histoPSet.jetPt.nbins = cms.int32 (15)
mssmHbbBtagTriggerMonitorAH200.histoPSet.jetPt.xmin  = cms.double(200)
mssmHbbBtagTriggerMonitorAH200.histoPSet.jetPt.xmax  = cms.double(350)
mssmHbbBtagTriggerMonitorAH200.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets200_v*')

mssmHbbBtagTriggerMonitorAH350 = mssmHbbBtagTriggerMonitor.clone()
mssmHbbBtagTriggerMonitorAH350.dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt350")
mssmHbbBtagTriggerMonitorAH350.jetPtMin = cms.double(350)
mssmHbbBtagTriggerMonitorAH350.triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching")
mssmHbbBtagTriggerMonitorAH350.histoPSet.jetPt.nbins = cms.int32 (25)
mssmHbbBtagTriggerMonitorAH350.histoPSet.jetPt.xmin  = cms.double(350)
mssmHbbBtagTriggerMonitorAH350.histoPSet.jetPt.xmax  = cms.double(600)
mssmHbbBtagTriggerMonitorAH350.genericTriggerEventPSet.hltPaths = cms.vstring('HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets350_v*')
