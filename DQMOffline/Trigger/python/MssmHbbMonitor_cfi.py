import FWCore.ParameterSet.Config as cms

mssmHbbBtagTriggerMonitorSL40noMu = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40_noMuon"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets30_SingleBTagCSV_0p92_DoublePFJets40_v"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)


mssmHbbBtagTriggerMonitorSL40 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets40_v"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)

mssmHbbBtagTriggerMonitorSL100 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt100"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets100_v"),
    jetPtMin = cms.double(100),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)

mssmHbbBtagTriggerMonitorSL200 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt200"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets200_v"),
    jetPtMin = cms.double(200),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)

mssmHbbBtagTriggerMonitorSL350 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt350"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets350_v"),
    jetPtMin = cms.double(350),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)


mssmHbbBtagTriggerMonitorAH100 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt100"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets100_v"),
    jetPtMin = cms.double(100),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)

mssmHbbBtagTriggerMonitorAH200 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt200"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets200_v"),
    jetPtMin = cms.double(200),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)

mssmHbbBtagTriggerMonitorAH350 = cms.EDAnalyzer(
    "TagAndProbeBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt350"),
    processname = cms.string("HLT"),
    pathname = cms.string("HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets350_v"),
    jetPtMin = cms.double(350),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    offlineBtag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    histoPSet = cms.PSet(
       jetPt = cms.PSet (
          nbins = cms.int32 (60),
          xmin  = cms.double(0),
          xmax  = cms.double(600),
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
          nbins = cms.int32 (40),
          xmin  = cms.double(-1),
          xmax  = cms.double(1),
       ),
    ),
)
