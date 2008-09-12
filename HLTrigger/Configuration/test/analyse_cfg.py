import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('anal1', 'anal2'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

# process.Timing = cms.Service("Timing")

# process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#     ignoreTotal = cms.untracked.int32(-1) ## default is one
# )

# process.Tracer = cms.Service("Tracer")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:HLTFromPureRaw.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false
)


import HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi
process.tsaAOD = HLTrigger.HLTcore.triggerSummaryAnalyzerAOD_cfi.triggerSummaryAnalyzerAOD.clone()
import HLTrigger.HLTcore.triggerSummaryAnalyzerRAW_cfi
process.tsaRAW = HLTrigger.HLTcore.triggerSummaryAnalyzerRAW_cfi.triggerSummaryAnalyzerRAW.clone()
import HLTrigger.HLTanalyzers.hlTrigReport_cfi
process.hltReport = HLTrigger.HLTanalyzers.hlTrigReport_cfi.hlTrigReport.clone()
process.tsa = cms.Path(process.tsaAOD+process.tsaRAW+process.hltReport)


process.aom = cms.OutputModule("AsciiOutputModule")

process.anal1 = cms.EDFilter("HLTAnalFilt",
    inputTag = cms.InputTag("hlt1jet400")
)

process.eca = cms.EDAnalyzer("EventContentAnalyzer")

import HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi
process.hltAnalyzerAOD = HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi.hltEventAnalyzerAOD.clone()
import HLTrigger.HLTcore.hltEventAnalyzerRAW_cfi
process.hltAnalyzerRAW = HLTrigger.HLTcore.hltEventAnalyzerRAW_cfi.hltEventAnalyzerRAW.clone()

process.final = cms.EndPath(process.hltAnalyzerAOD+process.hltAnalyzerRAW+process.aom+process.anal1+process.eca)
