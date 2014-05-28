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
    etTotalSource = cms.InputTag("l1tCaloUpgradeToGCTConverter"),
    nonIsolatedEmSource = cms.InputTag("l1tCaloUpgradeToGCTConverter","nonIsoEm"),
    etMissSource = cms.InputTag("l1tCaloUpgradeToGCTConverter"),
    htMissSource = cms.InputTag("l1tCaloUpgradeToGCTConverter"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("l1tCaloUpgradeToGCTConverter","forJets"),
    centralJetSource = cms.InputTag("l1tCaloUpgradeToGCTConverter","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("l1tCaloUpgradeToGCTConverter","tauJets"),
    isolatedEmSource = cms.InputTag("l1tCaloUpgradeToGCTConverter","isoEm"),
    etHadSource = cms.InputTag("l1tCaloUpgradeToGCTConverter"),
    hfRingEtSumsSource = cms.InputTag("l1tCaloUpgradeToGCTConverter"),
    hfRingBitCountsSource = cms.InputTag("l1tCaloUpgradeToGCTConverter"),
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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

readFiles = cms.untracked.vstring("file:SimL1Emulator_Stage1.root")
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

