import FWCore.ParameterSet.Config as cms

# make L1 ntuples from output of emulator

process = cms.Process("L1NTUPLE")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# output file
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('L1Tree.root')
)

# L1 ntuple producers
process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducer_Stage1Layer2_cfi")
process.load("L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi")

process.l1extraParticles = cms.EDProducer("L1ExtraParticlesProd",
    muonSource = cms.InputTag("simGtDigis"),
    etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm"),
    etMissSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    htMissSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets"),
    centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets"),
    isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm"),
    etHadSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    centralBxOnly = cms.bool(True),
    ignoreHtMiss = cms.bool(False)
)

process.p = cms.Path(
    process.l1NtupleProducer
    +process.l1extraParticles
    +process.l1ExtraTreeProducer
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
process.GlobalTag.globaltag = cms.string('POSTLS162_V2::All')
# For HI data
#process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V27A::All', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

readFiles = cms.untracked.vstring("file:SimL1Emulator_Stage1_PP.root")
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

