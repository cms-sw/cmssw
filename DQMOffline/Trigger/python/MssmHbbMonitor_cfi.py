import FWCore.ParameterSet.Config as cms

# physics path monitor

from DQMOffline.Trigger.mssmhbb_cfi import mssmHbbPhysicsMonitor

# physics path monitor
msssHbbAllHadronic100 = mssmHbbPhysicsMonitor.clone()
msssHbbAllHadronic100.dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/Physics/DoubleBJet100")
msssHbbAllHadronic100.pathname = cms.string("HLT_DoubleJets100_DoubleBTagCSV_0p92_DoublePFJets100MaxDeta1p6_v")

msssHbbAllHadronic116 = mssmHbbPhysicsMonitor.clone()
msssHbbAllHadronic116.dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/Physics/DoubleBJet116")
msssHbbAllHadronic116.pathname = cms.string("HLT_DoubleJets100_DoubleBTagCSV_0p92_DoublePFJets116MaxDeta1p6_v")

msssHbbAllHadronic128 = mssmHbbPhysicsMonitor.clone()
msssHbbAllHadronic128.dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/Physics/DoubleBJet128")
msssHbbAllHadronic128.pathname = cms.string("HLT_DoubleJets100_DoubleBTagCSV_0p92_DoublePFJets128MaxDeta1p6_v")

msssHbbSemileptonic40 = mssmHbbPhysicsMonitor.clone()
msssHbbSemileptonic40.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/Physics/Mu12DoubleBJet40")
msssHbbSemileptonic40.pathname = cms.string("HLT_DoubleJets30_Mu12_DoubleBTagCSV_0p92_DoublePFJets40MaxDeta1p6_v")

msssHbbSemileptonic54 = mssmHbbPhysicsMonitor.clone()
msssHbbSemileptonic54.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/Physics/Mu12DoubleBJet54")
msssHbbSemileptonic54.pathname = cms.string("HLT_DoubleJets30_Mu12_DoubleBTagCSV_0p92_DoublePFJets54MaxDeta1p6_v")

msssHbbSemileptonic62 = mssmHbbPhysicsMonitor.clone()
msssHbbSemileptonic62.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/Physics/Mu12DoubleBJet62")
msssHbbSemileptonic62.pathname = cms.string("HLT_DoubleJets30_Mu12_DoubleBTagCSV_0p92_DoublePFJets62MaxDeta1p6_v")

msssHbbSemileptonicNoBtag = mssmHbbPhysicsMonitor.clone()
msssHbbSemileptonicNoBtag.dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/Physics/Mu12Jet40")
msssHbbSemileptonicNoBtag.pathname = cms.string("HLT_SingleJet30_Mu12_SinglePFJet40_v")



# online btagging monitor
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
          nbins = cms.int32 (10),
          xmin  = cms.double(100),
          xmax  = cms.double(200),
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
          nbins = cms.int32 (15),
          xmin  = cms.double(200),
          xmax  = cms.double(350),
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
          nbins = cms.int32 (25),
          xmin  = cms.double(350),
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
          nbins = cms.int32 (20),
          xmin  = cms.double(0.8),
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
          nbins = cms.int32 (10),
          xmin  = cms.double(100),
          xmax  = cms.double(200),
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
          nbins = cms.int32 (15),
          xmin  = cms.double(200),
          xmax  = cms.double(350),
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
          nbins = cms.int32 (25),
          xmin  = cms.double(350),
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
          nbins = cms.int32 (20),
          xmin  = cms.double(0.8),
          xmax  = cms.double(1),
       ),
    ),
)
