import FWCore.ParameterSet.Config as cms

process = cms.Process("JETVALIDATION")


# INPUT
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source(
    "PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('file:FastJetReco.root')
    )

# OUTPUT
process.fileSaver = cms.EDFilter(
    "JetFileSaver",
    OutputFile = cms.untracked.string('ref.root')
    )

# MESSAGELOGGER
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

# DQMSTORE
process.DQMStore = cms.Service("DQMStore")


# IC5 Calo-Jets
process.ic5calo = cms.EDAnalyzer(
    "CaloJetTester",
    src = cms.InputTag("iterativeCone5CaloJets::RECO"),
    srcGen = cms.InputTag("iterativeCone5GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

# IC5 PFlow-Jets
process.ic5pflow = cms.EDFilter(
    "PFJetTester",
    src = cms.InputTag("iterativeCone5PFJets::RECO"),
    srcGen = cms.InputTag("iterativeCone5GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# KT4 Calo jets
process.kt4calo = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("kt4CaloJets::RECO"),
    srcGen = cms.InputTag("kt4GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )
                                    
# KT6 Calo jets
process.kt6calo = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("kt6CaloJets::RECO"),
    srcGen = cms.InputTag("kt6GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )
                                    
# SC5 Calo-Jets
process.sc5calo = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("sisCone5CaloJets::RECO"),
    srcGen = cms.InputTag("sisCone5GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

# SC7 Calo-Jets
process.sc7calo = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("sisCone7CaloJets::RECO"),
    srcGen = cms.InputTag("sisCone7GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

process.p1 = cms.Path(process.fileSaver+
                      process.ic5calo+process.ic5pflow+
                      process.kt4calo+process.kt6calo+
                      process.sc5calo+process.sc7calo
                      )

