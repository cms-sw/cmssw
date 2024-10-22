import FWCore.ParameterSet.Config as cms

process = cms.Process("REPORT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 999999999
process.MessageLogger.HLTrigReport=dict()

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/build/fwyzard/data/NanoDST/152658/0E2250DA-2CFB-DF11-90A8-001617C3B6FE.root',
        'file:/build/fwyzard/data/NanoDST/152658/42DAD707-2CFB-DF11-B008-001D09F28F25.root',
        'file:/build/fwyzard/data/NanoDST/152658/4C7D36D0-2FFB-DF11-A8E0-001D09F23A20.root',
        'file:/build/fwyzard/data/NanoDST/152658/7A3212D0-2FFB-DF11-8D74-001D09F253C0.root',
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.load( "HLTrigger.HLTanalyzers.hlTrigReport_cfi" )
process.hlTrigReport.HLTriggerResults   = cms.InputTag("TriggerResults", "", "HLT")
process.hlTrigReport.ReferencePath      = cms.untracked.string( "HLTriggerFinalPath" )
process.hlTrigReport.ReferenceRate      = cms.untracked.double( 100.0 )
process.hlTrigReport.ReportEvery        = "lumi"

process.report = cms.EndPath( process.hlTrigReport )
