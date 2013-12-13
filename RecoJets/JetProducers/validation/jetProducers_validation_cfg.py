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
    destinations = cms.untracked.vstring('cout'),
    cout         = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
)

# DQMSTORE
process.DQMStore = cms.Service("DQMStore")


# AK4 Calo-Jets
process.ak4calo = cms.EDAnalyzer(
    "CaloJetTester",
    src = cms.InputTag("ak4CaloJets::RECO"),
    srcGen = cms.InputTag("ak4GenJets::HLT"),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
    )

# AK4 PFlow-Jets
process.ak4pflow = cms.EDFilter(
    "PFJetTester",
    src = cms.InputTag("ak4PFJets::RECO"),
    srcGen = cms.InputTag("ak4GenJets::HLT"),
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
                      process.ak4calo+process.ak4pflow+
                      process.kt4calo+process.kt6calo+
                      process.sc5calo+process.sc7calo
                      )

