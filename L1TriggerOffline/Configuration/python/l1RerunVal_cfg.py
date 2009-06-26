import FWCore.ParameterSet.Config as cms

process = cms.Process("L1VAL")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_31X::All"

process.load("L1Trigger.Configuration.SimL1Emulator_cff")
import L1Trigger.Configuration.L1Extra_cff
process.hltL1extraParticles = L1Trigger.Configuration.L1Extra_cff.l1extraParticles.clone()
process.hltL1extraParticles.muonSource = cms.InputTag("simGmtDigis")
process.hltL1extraParticles.isolatedEmSource = cms.InputTag("simGctDigis","isoEm")
process.hltL1extraParticles.nonIsolatedEmSource = cms.InputTag("simGctDigis","nonIsoEm")
process.hltL1extraParticles.forwardJetSource = cms.InputTag("simGctDigis","forJets")
process.hltL1extraParticles.centralJetSource = cms.InputTag("simGctDigis","cenJets")
process.hltL1extraParticles.tauJetSource = cms.InputTag("simGctDigis","tauJets")
process.hltL1extraParticles.etMissSource = cms.InputTag("simGctDigis")
process.hltL1extraParticles.etTotalSource = cms.InputTag("simGctDigis")
process.hltL1extraParticles.htMissSource = cms.InputTag("simGctDigis")
process.hltL1extraParticles.etHadSource = cms.InputTag("simGctDigis")
process.hltL1extraParticles.hfRingEtSumsSource = cms.InputTag("simGctDigis")
process.hltL1extraParticles.hfRingBitCountsSource = cms.InputTag("simGctDigis")

process.load("L1TriggerOffline.L1Analyzer.L1MCAnalysis_cff")
process.p = cms.Path(
    process.pdigi
    +process.SimL1Emulator
    +process.hltL1ExtraParticles
    +process.L1MCAnalysis
)

# endpath
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag("simGtDigis")
process.l1GtTrigReport.L1GtRecordInputTag = 'simGtDigis'

process.e = cms.EndPath(process.l1GtTrigReport)

# output file
process.TFileService.fileName = 'l1RerunVal.root'

# input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:TTbar_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


