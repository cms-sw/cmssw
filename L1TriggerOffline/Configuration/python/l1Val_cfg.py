import FWCore.ParameterSet.Config as cms

process = cms.Process("l1validation")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# L1Extra (in case required in future)
process.load("L1Trigger.Configuration.L1Extra_cff")
process.l1extraParticles.muonSource = cms.InputTag("hltGtDigis")
process.l1extraParticles.isolatedEmSource = cms.InputTag("hltGctDigis","isoEm")
process.l1extraParticles.nonIsolatedEmSource = cms.InputTag("hltGctDigis","nonIsoEm")
process.l1extraParticles.forwardJetSource = cms.InputTag("hltGctDigis","forJets")
process.l1extraParticles.centralJetSource = cms.InputTag("hltGctDigis","cenJets")
process.l1extraParticles.tauJetSource = cms.InputTag("hltGctDigis","tauJets")
process.l1extraParticles.etTotalSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.etHadSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.etMissSource = cms.InputTag("hltGctDigis")

# path
process.load("L1TriggerOffline.L1Analyzer.L1MCAnalysis_cff")
process.p = cms.Path(
    process.L1MCAnalysis
)

# endpath
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag("hltGtDigis")

process.e = cms.EndPath(process.l1GtTrigReport)

# output file
process.TFileService.fileName = 'l1Val.root'

# input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:TTbar_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
