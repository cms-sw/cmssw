import FWCore.ParameterSet.Config as cms

mssmHbbBtagTriggerMonitorSL40noMu = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40_noMuon"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets30_SingleBTagCSV_0p92_DoublePFJets40_v"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)


mssmHbbBtagTriggerMonitorSL40 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt40"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets40_v"),
    jetPtMin = cms.double(40),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)

mssmHbbBtagTriggerMonitorSL100 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt100"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets100_v"),
    jetPtMin = cms.double(100),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)

mssmHbbBtagTriggerMonitorSL200 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt200"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets200_v"),
    jetPtMin = cms.double(200),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)

mssmHbbBtagTriggerMonitorSL350 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Semilep/BtagTrigger/pt350"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets30_Mu12_SingleBTagCSV_0p92_DoublePFJets350_v"),
    jetPtMin = cms.double(350),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)


mssmHbbBtagTriggerMonitorAH100 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt100"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets100_v"),
    jetPtMin = cms.double(100),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)

mssmHbbBtagTriggerMonitorAH200 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt200"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets200_v"),
    jetPtMin = cms.double(200),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)

mssmHbbBtagTriggerMonitorAH350 = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/Allhad/BtagTrigger/pt350"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets100_SingleBTagCSV_0p92_DoublePFJets350_v"),
    jetPtMin = cms.double(350),
    jetEtaMax = cms.double(2.2),
    tagBtagMin = cms.double(0.95),
    probeBtagMin = cms.double(0.84),
    triggerobjbtag = cms.string("hltBTagCalo80x6CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)
