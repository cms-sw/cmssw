import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.TriggerSummaryAnalyzerAOD=dict()
process.MessageLogger.TriggerSummaryAnalyzerRAW=dict()
process.MessageLogger.HLTEventAnalyzerAOD=dict()
process.MessageLogger.HLTEventAnalyzerRAW=dict()
process.MessageLogger.L1GtTrigReport=dict()
process.MessageLogger.L1TGlobalSummary=dict()
process.MessageLogger.HLTrigReport=dict()
process.MessageLogger.HLTSummaryFilter=dict()
process.MessageLogger.HLTConfigProvider=dict()
process.MessageLogger.HLTPrescaleProvider=dict()
process.MessageLogger.HLTConfigData=dict()

# process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.load('Configuration.StandardSequences.CondDBESSource_cff')
from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
# process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:run2_hlt_GRun')
process.GlobalTag = customiseGlobalTag(None, globaltag = 'auto:run2_hlt_GRun')

# process.Timing = cms.Service("Timing")

# process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#     ignoreTotal = cms.untracked.int32(-1) ## default is one
# )

# process.Tracer = cms.Service("Tracer")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:RelVal_HLT_GRun_DATA.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false
)


import HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi
process.tsaAOD = HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi.triggerSummaryAnalyzerAOD.clone()
import HLTrigger.HLTcore.triggerSummaryAnalyzerRAW_cfi
process.tsaRAW = HLTrigger.HLTcore.triggerSummaryAnalyzerRAW_cfi.triggerSummaryAnalyzerRAW.clone()
process.tsa = cms.Path(process.tsaAOD)#+process.tsaRAW)


import HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi
process.hltAOD = HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi.hltEventAnalyzerAOD.clone()
process.hltAOD.processName = cms.string("HLT")
process.hltAOD.triggerResults = cms.InputTag("TriggerResults","","HLT")
process.hltAOD.triggerEvent   = cms.InputTag("hltTriggerSummaryAOD","","HLT")

import HLTrigger.HLTcore.hltEventAnalyzerRAW_cfi
process.hltRAW = HLTrigger.HLTcore.hltEventAnalyzerRAW_cfi.hltEventAnalyzerRAW.clone()
process.hlt = cms.Path(process.hltAOD)#+process.hltRAW)


import HLTrigger.HLTanalyzers.hltTrigReport_cfi
process.hltReport = HLTrigger.HLTanalyzers.hltTrigReport_cfi.hltTrigReport.clone()
process.hltReport.HLTriggerResults = cms.InputTag("TriggerResults","","HLT")

process.aom = cms.OutputModule("AsciiOutputModule")
process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.final = cms.EndPath(process.hltReport+process.aom)#+process.eca)
