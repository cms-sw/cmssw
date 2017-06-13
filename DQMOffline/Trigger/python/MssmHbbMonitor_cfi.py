import FWCore.ParameterSet.Config as cms

mssmHbbBtagTriggerMonitor = cms.EDAnalyzer(
    "MssmHbbBtagTriggerMonitor",
    dirname = cms.string("HLT/Higgs/MssmHbb/BtagTrigger"),
    processname = cms.string("reHLT"),
    pathname = cms.string("HLT_DoubleJets30_SingleBTagCSV_0p92_DoublePFJets40_v"),
    triggerobjbtag = cms.string("hltBTagCalo30x8CSVp0p92SingleWithMatching"),
    triggerSummary = cms.InputTag("hltTriggerSummaryAOD","","reHLT"),
    triggerResults = cms.InputTag("TriggerResults","","reHLT"),
    offlineCSVPF = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    
)
