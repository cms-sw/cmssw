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
    etTotalSource = cms.InputTag("Layer2gctFormat"),
    nonIsolatedEmSource = cms.InputTag("Layer2gctFormat","nonIsoEm"),
    etMissSource = cms.InputTag("Layer2gctFormat"),
    htMissSource = cms.InputTag("Layer2gctFormat"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("Layer2gctFormat","forJets"),
    centralJetSource = cms.InputTag("Layer2gctFormat","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("Layer2gctFormat","tauJets"),
    isolatedEmSource = cms.InputTag("Layer2gctFormat","isoEm"),
    etHadSource = cms.InputTag("Layer2gctFormat"),
    hfRingEtSumsSource = cms.InputTag("Layer2gctFormat"),
    hfRingBitCountsSource = cms.InputTag("Layer2gctFormat"),
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
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V27A::All', '')

#SkipEvent = cms.untracked.vstring('ProductNotFound')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

readFiles = cms.untracked.vstring("file:L1Emulator_HI_newLayer2.root")
secFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

