# Conditions read from  CMS_CONDITIONS  via FrontierProd 
import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_4_0_pre8/RelValProdTTbar_13/AODSIM/MCRUN2_74_V7-v1/00000/44E1E4BA-50BD-E411-A57A-002618943949.root')
)
process.AK5PFCandMETcorr = cms.EDProducer("PFCandMETcorrInputProducer",
    src = cms.InputTag("AK5PFpfCandsNotInJetsForMetCorr")
)


process.AK5PFCombinedCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag(cms.InputTag("ak5PFL1FastjetCorrector"), cms.InputTag("ak5PFL2RelativeCorrector"), cms.InputTag("ak5PFL3AbsoluteCorrector"))
)


process.AK5PFJetMETcorr = cms.EDProducer("PFJetMETcorrInputProducer",
    jetCorrEtaMax = cms.double(9.9),
    jetCorrLabel = cms.InputTag("AK5PFCombinedCorrector"),
    offsetCorrLabel = cms.InputTag("AK5PFL1FastJet"),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.9),
    skipMuonSelection = cms.string('isGlobalMuon | isStandAloneMuon'),
    skipMuons = cms.bool(True),
    src = cms.InputTag("ak4PFJets"),
    type1JetPtThreshold = cms.double(10.0)
)


process.AK5PFL1FastJet = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.AK5PFL1Offset = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.AK5PFL2L3Residual = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.AK5PFL2Relative = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L2Relative')
)


process.AK5PFL3Absolute = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L3Absolute')
)


process.AK5PFType1CorMet = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("AK5PFJetMETcorr","type1"))
)


process.AK5PFType1p2CorMet = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("AK5PFJetMETcorr","type1"), "AK5PFcorrPfMetType2")
)


process.AK5PFcorrPfMetType2 = cms.EDProducer("Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(cms.InputTag("AK5PFJetMETcorr","type2"), cms.InputTag("AK5PFJetMETcorr","offset"), cms.InputTag("AK5PFCandMETcorr")),
    type2CorrFormula = cms.string('A'),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.4)
    )
)


process.AK5PFpfCandsNotInJetsForMetCorr = cms.EDProducer("PFCandidateFromFwdPtrProducer",
    src = cms.InputTag("AK5PFpfCandsNotInJetsPtrForMetCorr")
)


process.AK5PFpfCandsNotInJetsPtrForMetCorr = cms.EDProducer("TPPFJetsOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlowPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('noJet'),
    topCollection = cms.InputTag("AK5PFpfJetsPtrForMetCorr"),
    verbose = cms.untracked.bool(False)
)


process.AK5PFpfJetsPtrForMetCorr = cms.EDProducer("PFJetFwdPtrProducer",
    src = cms.InputTag("ak4PFJets")
)


process.Njettiness = cms.EDProducer("NjettinessAdder",
    Njets = cms.vuint32(1, 2, 3, 4),
    R0 = cms.double(0.8),
    Rcutoff = cms.double(-999.0),
    akAxesR0 = cms.double(-999.0),
    axesDefinition = cms.uint32(6),
    beta = cms.double(1.0),
    measureDefinition = cms.uint32(0),
    nPass = cms.int32(-999),
    src = cms.InputTag("ak8PFJetsCHS")
)


process.NjettinessAK8 = cms.EDProducer("NjettinessAdder",
    Njets = cms.vuint32(1, 2, 3, 4),
    R0 = cms.double(0.8),
    Rcutoff = cms.double(-999.0),
    akAxesR0 = cms.double(-999.0),
    axesDefinition = cms.uint32(6),
    beta = cms.double(1.0),
    cone = cms.double(0.8),
    measureDefinition = cms.uint32(0),
    nPass = cms.int32(-999),
    src = cms.InputTag("ak8PFJetsCHS")
)


process.NjettinessCA8 = cms.EDProducer("NjettinessAdder",
    Njets = cms.vuint32(1, 2, 3, 4),
    R0 = cms.double(0.8),
    Rcutoff = cms.double(-999.0),
    akAxesR0 = cms.double(-999.0),
    axesDefinition = cms.uint32(6),
    beta = cms.double(1.0),
    cone = cms.double(0.8),
    measureDefinition = cms.uint32(0),
    nPass = cms.int32(-999),
    src = cms.InputTag("ca8PFJetsCHS")
)


process.QGTagger = cms.EDProducer("QGTagger",
    jetsLabel = cms.string('QGL_AK4PFchs'),
    srcJets = cms.InputTag("ak4PFJetsCHS"),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll"),
    srcVertexCollection = cms.InputTag("offlinePrimaryVerticesWithBS"),
    useQualityCuts = cms.bool(False)
)


process.QJetsAdder = cms.EDProducer("QjetsAdder",
    cutoff = cms.double(10.0),
    dcutfctr = cms.double(0.5),
    expmax = cms.double(0.0),
    expmin = cms.double(0.0),
    jetAlgo = cms.string('CA'),
    jetRad = cms.double(0.8),
    ntrial = cms.int32(50),
    preclustering = cms.int32(50),
    rigidity = cms.double(0.1),
    src = cms.InputTag("ak8PFJetsCHS"),
    zcut = cms.double(0.1)
)


process.QJetsAdderAK8 = cms.EDProducer("QjetsAdder",
    cutoff = cms.double(10.0),
    dcutfctr = cms.double(0.5),
    expmax = cms.double(0.0),
    expmin = cms.double(0.0),
    jetAlgo = cms.string('AK'),
    jetRad = cms.double(0.8),
    ntrial = cms.int32(50),
    preclustering = cms.int32(50),
    rigidity = cms.double(0.1),
    src = cms.InputTag("ak8PFJetsCHS"),
    zcut = cms.double(0.1)
)


process.QJetsAdderCA8 = cms.EDProducer("QjetsAdder",
    cutoff = cms.double(10.0),
    dcutfctr = cms.double(0.5),
    expmax = cms.double(0.0),
    expmin = cms.double(0.0),
    jetAlgo = cms.string('CA'),
    jetRad = cms.double(0.8),
    ntrial = cms.int32(50),
    preclustering = cms.int32(50),
    rigidity = cms.double(0.1),
    src = cms.InputTag("ca8PFJetsCHS"),
    zcut = cms.double(0.1)
)


process.ak10PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak10PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFCHSL1FastjetCorrector", "ak10PFCHSL2RelativeCorrector", "ak10PFCHSL3AbsoluteCorrector", "ak10PFCHSResidualCorrector")
)


process.ak10PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFCHSL1OffsetCorrector", "ak10PFCHSL2RelativeCorrector", "ak10PFCHSL3AbsoluteCorrector", "ak10PFCHSResidualCorrector")
)


process.ak10PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak10PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFCHSL2RelativeCorrector", "ak10PFCHSL3AbsoluteCorrector")
)


process.ak10PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFCHSL2RelativeCorrector", "ak10PFCHSL3AbsoluteCorrector", "ak10PFCHSResidualCorrector")
)


process.ak10PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L2Relative')
)


process.ak10PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L3Absolute')
)


process.ak10PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak10PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak10PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFL1FastjetCorrector", "ak10PFL2RelativeCorrector", "ak10PFL3AbsoluteCorrector", "ak10PFResidualCorrector")
)


process.ak10PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFL1OffsetCorrector", "ak10PFL2RelativeCorrector", "ak10PFL3AbsoluteCorrector", "ak10PFResidualCorrector")
)


process.ak10PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak10PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFL2RelativeCorrector", "ak10PFL3AbsoluteCorrector")
)


process.ak10PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak10PFL2RelativeCorrector", "ak10PFL3AbsoluteCorrector", "ak10PFResidualCorrector")
)


process.ak10PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L2Relative')
)


process.ak10PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L3Absolute')
)


process.ak10PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak1HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.1),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak1PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak1PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFCHSL1FastjetCorrector", "ak1PFCHSL2RelativeCorrector", "ak1PFCHSL3AbsoluteCorrector", "ak1PFCHSResidualCorrector")
)


process.ak1PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFCHSL1OffsetCorrector", "ak1PFCHSL2RelativeCorrector", "ak1PFCHSL3AbsoluteCorrector", "ak1PFCHSResidualCorrector")
)


process.ak1PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak1PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFCHSL2RelativeCorrector", "ak1PFCHSL3AbsoluteCorrector")
)


process.ak1PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFCHSL2RelativeCorrector", "ak1PFCHSL3AbsoluteCorrector", "ak1PFCHSResidualCorrector")
)


process.ak1PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L2Relative')
)


process.ak1PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L3Absolute')
)


process.ak1PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak1PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak1PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFL1FastjetCorrector", "ak1PFL2RelativeCorrector", "ak1PFL3AbsoluteCorrector", "ak1PFResidualCorrector")
)


process.ak1PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFL1OffsetCorrector", "ak1PFL2RelativeCorrector", "ak1PFL3AbsoluteCorrector", "ak1PFResidualCorrector")
)


process.ak1PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak1PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFL2RelativeCorrector", "ak1PFL3AbsoluteCorrector")
)


process.ak1PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak1PFL2RelativeCorrector", "ak1PFL3AbsoluteCorrector", "ak1PFResidualCorrector")
)


process.ak1PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L2Relative')
)


process.ak1PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L3Absolute')
)


process.ak1PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak2HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.2),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak2PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak2PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFCHSL1FastjetCorrector", "ak2PFCHSL2RelativeCorrector", "ak2PFCHSL3AbsoluteCorrector", "ak2PFCHSResidualCorrector")
)


process.ak2PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFCHSL1OffsetCorrector", "ak2PFCHSL2RelativeCorrector", "ak2PFCHSL3AbsoluteCorrector", "ak2PFCHSResidualCorrector")
)


process.ak2PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak2PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFCHSL2RelativeCorrector", "ak2PFCHSL3AbsoluteCorrector")
)


process.ak2PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFCHSL2RelativeCorrector", "ak2PFCHSL3AbsoluteCorrector", "ak2PFCHSResidualCorrector")
)


process.ak2PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L2Relative')
)


process.ak2PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L3Absolute')
)


process.ak2PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak2PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak2PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFL1FastjetCorrector", "ak2PFL2RelativeCorrector", "ak2PFL3AbsoluteCorrector", "ak2PFResidualCorrector")
)


process.ak2PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFL1OffsetCorrector", "ak2PFL2RelativeCorrector", "ak2PFL3AbsoluteCorrector", "ak2PFResidualCorrector")
)


process.ak2PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak2PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFL2RelativeCorrector", "ak2PFL3AbsoluteCorrector")
)


process.ak2PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak2PFL2RelativeCorrector", "ak2PFL3AbsoluteCorrector", "ak2PFResidualCorrector")
)


process.ak2PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L2Relative')
)


process.ak2PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L3Absolute')
)


process.ak2PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak3HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.3),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak3PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak3PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFCHSL1FastjetCorrector", "ak3PFCHSL2RelativeCorrector", "ak3PFCHSL3AbsoluteCorrector", "ak3PFCHSResidualCorrector")
)


process.ak3PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFCHSL1OffsetCorrector", "ak3PFCHSL2RelativeCorrector", "ak3PFCHSL3AbsoluteCorrector", "ak3PFCHSResidualCorrector")
)


process.ak3PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak3PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFCHSL2RelativeCorrector", "ak3PFCHSL3AbsoluteCorrector")
)


process.ak3PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFCHSL2RelativeCorrector", "ak3PFCHSL3AbsoluteCorrector", "ak3PFCHSResidualCorrector")
)


process.ak3PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L2Relative')
)


process.ak3PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L3Absolute')
)


process.ak3PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak3PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak3PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFL1FastjetCorrector", "ak3PFL2RelativeCorrector", "ak3PFL3AbsoluteCorrector", "ak3PFResidualCorrector")
)


process.ak3PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFL1OffsetCorrector", "ak3PFL2RelativeCorrector", "ak3PFL3AbsoluteCorrector", "ak3PFResidualCorrector")
)


process.ak3PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak3PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFL2RelativeCorrector", "ak3PFL3AbsoluteCorrector")
)


process.ak3PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak3PFL2RelativeCorrector", "ak3PFL3AbsoluteCorrector", "ak3PFResidualCorrector")
)


process.ak3PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L2Relative')
)


process.ak3PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L3Absolute')
)


process.ak3PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak4CaloL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector")
)


process.ak4CaloL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector", "ak4CaloL6SLBCorrector")
)


process.ak4CaloL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector", "ak4CaloResidualCorrector")
)


process.ak4CaloL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak4CaloL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1OffsetCorrector", "ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector")
)


process.ak4CaloL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1OffsetCorrector", "ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector", "ak4CaloResidualCorrector")
)


process.ak4CaloL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak4CaloL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector")
)


process.ak4CaloL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector", "ak4CaloL6SLBCorrector")
)


process.ak4CaloL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL2RelativeCorrector", "ak4CaloL3AbsoluteCorrector", "ak4CaloResidualCorrector")
)


process.ak4CaloL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2Relative')
)


process.ak4CaloL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L3Absolute')
)


process.ak4CaloL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak4CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak4CaloJetsSoftMuonTagInfos")
)


process.ak4CaloResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.ak4GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak4GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak4GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak4HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak4JPTL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4JPTL1FastjetCorrector", "ak4JPTL2RelativeCorrector", "ak4JPTL3AbsoluteCorrector")
)


process.ak4JPTL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4JPTL1FastjetCorrector", "ak4JPTL2RelativeCorrector", "ak4JPTL3AbsoluteCorrector", "ak4JPTResidualCorrector")
)


process.ak4JPTL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak4JPTL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4L1JPTOffsetCorrector", "ak4JPTL2RelativeCorrector", "ak4JPTL3AbsoluteCorrector")
)


process.ak4JPTL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4L1JPTOffsetCorrector", "ak4JPTL2RelativeCorrector", "ak4JPTL3AbsoluteCorrector", "ak4JPTResidualCorrector")
)


process.ak4JPTL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4JPT'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak4JPTL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4L1JPTOffsetCorrector", "ak4JPTL2RelativeCorrector", "ak4JPTL3AbsoluteCorrector")
)


process.ak4JPTL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4L1JPTOffsetCorrector", "ak4JPTL2RelativeCorrector", "ak4JPTL3AbsoluteCorrector", "ak4JPTResidualCorrector")
)


process.ak4JPTL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4JPT'),
    level = cms.string('L2Relative')
)


process.ak4JPTL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4JPT'),
    level = cms.string('L3Absolute')
)


process.ak4JPTResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4JPT'),
    level = cms.string('L2L3Residual')
)


process.ak4L1JPTOffsetCorrector = cms.EDProducer("L1JPTOffsetCorrectorProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L1JPTOffset'),
    offsetService = cms.InputTag("ak4CaloL1OffsetCorrector")
)


process.ak4PFCHSL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL1FastjetCorrector", "ak4PFCHSL2RelativeCorrector", "ak4PFCHSL3AbsoluteCorrector")
)


process.ak4PFCHSL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL1FastjetCorrector", "ak4PFCHSL2RelativeCorrector", "ak4PFCHSL3AbsoluteCorrector", "ak4PFCHSResidualCorrector")
)


process.ak4PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak4PFCHSL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL1OffsetCorrector", "ak4PFCHSL2RelativeCorrector", "ak4PFCHSL3AbsoluteCorrector")
)


process.ak4PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL1OffsetCorrector", "ak4PFCHSL2RelativeCorrector", "ak4PFCHSL3AbsoluteCorrector", "ak4PFCHSResidualCorrector")
)


process.ak4PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak4PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL2RelativeCorrector", "ak4PFCHSL3AbsoluteCorrector")
)


process.ak4PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL2RelativeCorrector", "ak4PFCHSL3AbsoluteCorrector", "ak4PFCHSResidualCorrector")
)


process.ak4PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2Relative')
)


process.ak4PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L3Absolute')
)


process.ak4PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak4PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak4PFJetsCHS = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak4PFJetsPuppi = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("puppi"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak4PFJetsSK = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("softKiller"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak4PFL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector")
)


process.ak4PFL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector", "ak4PFL6SLBCorrector")
)


process.ak4PFL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector", "ak4PFResidualCorrector")
)


process.ak4PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak4PFL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1OffsetCorrector", "ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector")
)


process.ak4PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1OffsetCorrector", "ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector", "ak4PFResidualCorrector")
)


process.ak4PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak4PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector")
)


process.ak4PFL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector", "ak4PFL6SLBCorrector")
)


process.ak4PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL2RelativeCorrector", "ak4PFL3AbsoluteCorrector", "ak4PFResidualCorrector")
)


process.ak4PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2Relative')
)


process.ak4PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L3Absolute')
)


process.ak4PFL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak4PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak4PFJetsSoftMuonTagInfos")
)


process.ak4PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak4TrackL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "ak4TrackL2RelativeCorrector", "ak4TrackL3AbsoluteCorrector")
)


process.ak4TrackL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4TrackL2RelativeCorrector", "ak4TrackL3AbsoluteCorrector")
)


process.ak4TrackL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4TRK'),
    level = cms.string('L2Relative')
)


process.ak4TrackL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4TRK'),
    level = cms.string('L3Absolute')
)


process.ak5GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak5GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak5GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak5PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak5PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFCHSL1FastjetCorrector", "ak5PFCHSL2RelativeCorrector", "ak5PFCHSL3AbsoluteCorrector", "ak5PFCHSResidualCorrector")
)


process.ak5PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFCHSL1OffsetCorrector", "ak5PFCHSL2RelativeCorrector", "ak5PFCHSL3AbsoluteCorrector", "ak5PFCHSResidualCorrector")
)


process.ak5PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak5PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFCHSL2RelativeCorrector", "ak5PFCHSL3AbsoluteCorrector")
)


process.ak5PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFCHSL2RelativeCorrector", "ak5PFCHSL3AbsoluteCorrector", "ak5PFCHSResidualCorrector")
)


process.ak5PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L2Relative')
)


process.ak5PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L3Absolute')
)


process.ak5PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak5PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak5PFJetsCHS = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak5PFJetsCHSFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ak5PFJetsCHSPruned = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(2),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    rcut_factor = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    usePruning = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak5PFJetsCHSSoftDrop = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    R0 = cms.double(0.5),
    Rho_EtaMax = cms.double(4.4),
    beta = cms.double(0.0),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useSoftDrop = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak5PFJetsCHSTrimmed = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.2),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    trimPtFracMin = cms.double(0.03),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useTrimming = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak5PFJetsFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ak5PFJetsMassDropFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    muCut = cms.double(0.667),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    useMassDropTagger = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    yCut = cms.double(0.08)
)


process.ak5PFJetsPruned = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(2),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    rcut_factor = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    usePruning = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak5PFJetsSoftDrop = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    R0 = cms.double(0.5),
    Rho_EtaMax = cms.double(4.4),
    beta = cms.double(0.0),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useSoftDrop = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak5PFJetsTrimmed = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.2),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    trimPtFracMin = cms.double(0.03),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useTrimming = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak5PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak5PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFL1FastjetCorrector", "ak5PFL2RelativeCorrector", "ak5PFL3AbsoluteCorrector", "ak5PFResidualCorrector")
)


process.ak5PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFL1OffsetCorrector", "ak5PFL2RelativeCorrector", "ak5PFL3AbsoluteCorrector", "ak5PFResidualCorrector")
)


process.ak5PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak5PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFL2RelativeCorrector", "ak5PFL3AbsoluteCorrector")
)


process.ak5PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak5PFL2RelativeCorrector", "ak5PFL3AbsoluteCorrector", "ak5PFResidualCorrector")
)


process.ak5PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L2Relative')
)


process.ak5PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L3Absolute')
)


process.ak5PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak6HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak6PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak6PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFCHSL1FastjetCorrector", "ak6PFCHSL2RelativeCorrector", "ak6PFCHSL3AbsoluteCorrector", "ak6PFCHSResidualCorrector")
)


process.ak6PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFCHSL1OffsetCorrector", "ak6PFCHSL2RelativeCorrector", "ak6PFCHSL3AbsoluteCorrector", "ak6PFCHSResidualCorrector")
)


process.ak6PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak6PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFCHSL2RelativeCorrector", "ak6PFCHSL3AbsoluteCorrector")
)


process.ak6PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFCHSL2RelativeCorrector", "ak6PFCHSL3AbsoluteCorrector", "ak6PFCHSResidualCorrector")
)


process.ak6PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L2Relative')
)


process.ak6PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L3Absolute')
)


process.ak6PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak6PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak6PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFL1FastjetCorrector", "ak6PFL2RelativeCorrector", "ak6PFL3AbsoluteCorrector", "ak6PFResidualCorrector")
)


process.ak6PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFL1OffsetCorrector", "ak6PFL2RelativeCorrector", "ak6PFL3AbsoluteCorrector", "ak6PFResidualCorrector")
)


process.ak6PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak6PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFL2RelativeCorrector", "ak6PFL3AbsoluteCorrector")
)


process.ak6PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak6PFL2RelativeCorrector", "ak6PFL3AbsoluteCorrector", "ak6PFResidualCorrector")
)


process.ak6PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L2Relative')
)


process.ak6PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L3Absolute')
)


process.ak6PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak7CaloL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector")
)


process.ak7CaloL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL1OffsetCorrector", "ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector", "ak7CaloL6SLBCorrector")
)


process.ak7CaloL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL1FastjetCorrector", "ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector", "ak7CaloResidualCorrector")
)


process.ak7CaloL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak7CaloL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL1OffsetCorrector", "ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector")
)


process.ak7CaloL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL1OffsetCorrector", "ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector", "ak7CaloResidualCorrector")
)


process.ak7CaloL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak7CaloL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector")
)


process.ak7CaloL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector", "ak7CaloL6SLBCorrector")
)


process.ak7CaloL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7CaloL2RelativeCorrector", "ak7CaloL3AbsoluteCorrector", "ak7CaloResidualCorrector")
)


process.ak7CaloL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L2Relative')
)


process.ak7CaloL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L3Absolute')
)


process.ak7CaloL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak7CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak7CaloJetsSoftMuonTagInfos")
)


process.ak7CaloResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.ak7HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.7),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak7JPTL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7JPTL1FastjetCorrector", "ak7L1JPTOffsetCorrector", "ak7JPTL2RelativeCorrector", "ak7JPTL3AbsoluteCorrector", "ak7JPTResidualCorrector")
)


process.ak7JPTL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak7JPTL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7JPTL1OffsetCorrector", "ak7L1JPTOffsetCorrector", "ak7JPTL2RelativeCorrector", "ak7JPTL3AbsoluteCorrector")
)


process.ak7JPTL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7JPTL1OffsetCorrector", "ak7L1JPTOffsetCorrector", "ak7JPTL2RelativeCorrector", "ak7JPTL3AbsoluteCorrector", "ak7JPTResidualCorrector")
)


process.ak7JPTL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak7JPTL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7L1JPTOffsetCorrector", "ak7JPTL2RelativeCorrector", "ak7JPTL3AbsoluteCorrector")
)


process.ak7JPTL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7JPT'),
    level = cms.string('L2Relative')
)


process.ak7JPTL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7JPT'),
    level = cms.string('L3Absolute')
)


process.ak7JPTL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak7JPTJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak7JPTJetsSoftMuonTagInfos")
)


process.ak7JPTResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.ak7L1JPTOffsetCorrector = cms.EDProducer("L1JPTOffsetCorrectorProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L1JPTOffset'),
    offsetService = cms.InputTag("ak7CaloL1OffsetCorrector")
)


process.ak7PFCHSL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFCHSL1FastjetCorrector", "ak7PFCHSL2RelativeCorrector", "ak7PFCHSL3AbsoluteCorrector")
)


process.ak7PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak7PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFCHSL1FastjetCorrector", "ak7PFCHSL2RelativeCorrector", "ak7PFCHSL3AbsoluteCorrector", "ak7PFCHSResidualCorrector")
)


process.ak7PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFCHSL1OffsetCorrector", "ak7PFCHSL2RelativeCorrector", "ak7PFCHSL3AbsoluteCorrector", "ak7PFCHSResidualCorrector")
)


process.ak7PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak7PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFCHSL2RelativeCorrector", "ak7PFCHSL3AbsoluteCorrector")
)


process.ak7PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFCHSL2RelativeCorrector", "ak7PFCHSL3AbsoluteCorrector", "ak7PFCHSResidualCorrector")
)


process.ak7PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L2Relative')
)


process.ak7PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L3Absolute')
)


process.ak7PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak7PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.7),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak7PFL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector")
)


process.ak7PFL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector", "ak7PFL6SLBCorrector")
)


process.ak7PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak7PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFL1FastjetCorrector", "ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector", "ak7PFResidualCorrector")
)


process.ak7PFL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFL1OffsetCorrector", "ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector")
)


process.ak7PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFL1OffsetCorrector", "ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector", "ak7PFResidualCorrector")
)


process.ak7PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak7PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector")
)


process.ak7PFL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector", "ak7PFL6SLBCorrector")
)


process.ak7PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak7PFL2RelativeCorrector", "ak7PFL3AbsoluteCorrector", "ak7PFResidualCorrector")
)


process.ak7PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L2Relative')
)


process.ak7PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L3Absolute')
)


process.ak7PFL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak7PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak7PFJetsSoftMuonTagInfos")
)


process.ak7PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak8GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak8GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak8GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ak8PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak8PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFCHSL1FastjetCorrector", "ak8PFCHSL2RelativeCorrector", "ak8PFCHSL3AbsoluteCorrector", "ak8PFCHSResidualCorrector")
)


process.ak8PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFCHSL1OffsetCorrector", "ak8PFCHSL2RelativeCorrector", "ak8PFCHSL3AbsoluteCorrector", "ak8PFCHSResidualCorrector")
)


process.ak8PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak8PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFCHSL2RelativeCorrector", "ak8PFCHSL3AbsoluteCorrector")
)


process.ak8PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFCHSL2RelativeCorrector", "ak8PFCHSL3AbsoluteCorrector", "ak8PFCHSResidualCorrector")
)


process.ak8PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L2Relative')
)


process.ak8PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L3Absolute')
)


process.ak8PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak8PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak8PFJetsCHS = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(50.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak8PFJetsCHSFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ak8PFJetsCHSFilteredMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ak8PFJetsCHSFiltered"),
    src = cms.InputTag("ak8PFJetsCHS"),
    value = cms.string('mass')
)


process.ak8PFJetsCHSPruned = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(2),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    rcut_factor = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    usePruning = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak8PFJetsCHSPrunedMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ak8PFJetsCHSPruned"),
    src = cms.InputTag("ak8PFJetsCHS"),
    value = cms.string('mass')
)


process.ak8PFJetsCHSSoftDrop = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    R0 = cms.double(0.8),
    Rho_EtaMax = cms.double(4.4),
    beta = cms.double(0.0),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useSoftDrop = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak8PFJetsCHSSoftDropMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ak8PFJetsCHSSoftDrop"),
    src = cms.InputTag("ak8PFJetsCHS"),
    value = cms.string('mass')
)


process.ak8PFJetsCHSTrimmed = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.2),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    trimPtFracMin = cms.double(0.03),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useTrimming = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak8PFJetsCHSTrimmedMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ak8PFJetsCHSTrimmed"),
    src = cms.InputTag("ak8PFJetsCHS"),
    value = cms.string('mass')
)


process.ak8PFJetsCS = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    csRParam = cms.double(0.4),
    csRho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useConstituentSubtraction = cms.bool(True),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak8PFJetsCSFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("ak8PFJetsCSConstituents","constituents"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ak8PFJetsCSPruned = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(2),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    rcut_factor = cms.double(0.5),
    src = cms.InputTag("ak8PFJetsCSConstituents","constituents"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    usePruning = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ak8PFJetsCSTrimmed = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('AntiKt'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.2),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("ak8PFJetsCSConstituents","constituents"),
    srcPVs = cms.InputTag(""),
    trimPtFracMin = cms.double(0.03),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useTrimming = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ak8PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak8PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFL1FastjetCorrector", "ak8PFL2RelativeCorrector", "ak8PFL3AbsoluteCorrector", "ak8PFResidualCorrector")
)


process.ak8PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFL1OffsetCorrector", "ak8PFL2RelativeCorrector", "ak8PFL3AbsoluteCorrector", "ak8PFResidualCorrector")
)


process.ak8PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak8PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFL2RelativeCorrector", "ak8PFL3AbsoluteCorrector")
)


process.ak8PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak8PFL2RelativeCorrector", "ak8PFL3AbsoluteCorrector", "ak8PFResidualCorrector")
)


process.ak8PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L2Relative')
)


process.ak8PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L3Absolute')
)


process.ak8PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak9PFCHSL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak9PFCHSL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFCHSL1FastjetCorrector", "ak9PFCHSL2RelativeCorrector", "ak9PFCHSL3AbsoluteCorrector", "ak9PFCHSResidualCorrector")
)


process.ak9PFCHSL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFCHSL1OffsetCorrector", "ak9PFCHSL2RelativeCorrector", "ak9PFCHSL3AbsoluteCorrector", "ak9PFCHSResidualCorrector")
)


process.ak9PFCHSL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak9PFCHSL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFCHSL2RelativeCorrector", "ak9PFCHSL3AbsoluteCorrector")
)


process.ak9PFCHSL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFCHSL2RelativeCorrector", "ak9PFCHSL3AbsoluteCorrector", "ak9PFCHSResidualCorrector")
)


process.ak9PFCHSL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L2Relative')
)


process.ak9PFCHSL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L3Absolute')
)


process.ak9PFCHSResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak9PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak9PFL1FastjetL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFL1FastjetCorrector", "ak9PFL2RelativeCorrector", "ak9PFL3AbsoluteCorrector", "ak9PFResidualCorrector")
)


process.ak9PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFL1OffsetCorrector", "ak9PFL2RelativeCorrector", "ak9PFL3AbsoluteCorrector", "ak9PFResidualCorrector")
)


process.ak9PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ak9PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFL2RelativeCorrector", "ak9PFL3AbsoluteCorrector")
)


process.ak9PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak9PFL2RelativeCorrector", "ak9PFL3AbsoluteCorrector", "ak9PFResidualCorrector")
)


process.ak9PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L2Relative')
)


process.ak9PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L3Absolute')
)


process.ak9PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.bToCharmDecayVertexMerged = cms.EDProducer("BtoCharmDecayVertexMerger",
    maxPtreltomerge = cms.untracked.double(7777.0),
    minCosPAtomerge = cms.untracked.double(0.99),
    minDRUnique = cms.untracked.double(0.4),
    minvecSumIMifsmallDRUnique = cms.untracked.double(5.5),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    secondaryVertices = cms.InputTag("inclusiveSecondaryVerticesFiltered")
)


process.ca15PFJetsCHSFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(1.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ca15PFJetsCHSMassDropFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    muCut = cms.double(0.667),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(1.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    useMassDropTagger = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    yCut = cms.double(0.08)
)


process.ca4GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ca4GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ca4GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ca4PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ca8GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ca8GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ca8GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.ca8PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ca8PFJetsCHS = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(50.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ca8PFJetsCHSFiltered = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(3),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.3),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useFiltering = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ca8PFJetsCHSFilteredMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ca8PFJetsCHSFiltered"),
    src = cms.InputTag("ca8PFJetsCHS"),
    value = cms.string('mass')
)


process.ca8PFJetsCHSPruned = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nFilt = cms.int32(2),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    rcut_factor = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    usePruning = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ca8PFJetsCHSPrunedMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ca8PFJetsCHSPruned"),
    src = cms.InputTag("ca8PFJetsCHS"),
    value = cms.string('mass')
)


process.ca8PFJetsCHSSoftDrop = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    R0 = cms.double(0.8),
    Rho_EtaMax = cms.double(4.4),
    beta = cms.double(0.0),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('SubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useSoftDrop = cms.bool(True),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True),
    zcut = cms.double(0.1)
)


process.ca8PFJetsCHSSoftDropMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ca8PFJetsCHSSoftDrop"),
    src = cms.InputTag("ca8PFJetsCHS"),
    value = cms.string('mass')
)


process.ca8PFJetsCHSTrimmed = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetPtMin = cms.double(15.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rFilt = cms.double(0.2),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    trimPtFracMin = cms.double(0.03),
    useDeterministicSeed = cms.bool(True),
    useExplicitGhosts = cms.bool(True),
    useTrimming = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.ca8PFJetsCHSTrimmedMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("ca8PFJetsCHSTrimmed"),
    src = cms.InputTag("ca8PFJetsCHS"),
    value = cms.string('mass')
)


process.caTopTagInfos = cms.EDProducer("CATopJetTagger",
    MinMassMax = cms.double(200.0),
    MinMassMin = cms.double(0.0),
    TopMass = cms.double(173),
    TopMassMax = cms.double(250.0),
    TopMassMin = cms.double(0.0),
    WMass = cms.double(80.4),
    WMassMax = cms.double(200.0),
    WMassMin = cms.double(0.0),
    src = cms.InputTag("cmsTopTagPFJetsCHS"),
    verbose = cms.bool(False)
)


process.caloMetT1 = cms.EDProducer("CorrectedCaloMETProducer",
    src = cms.InputTag("caloMetM"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrCaloMetType1","type1"))
)


process.caloMetT1T2 = cms.EDProducer("CorrectedCaloMETProducer",
    src = cms.InputTag("caloMetM"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrCaloMetType1","type1"), cms.InputTag("corrCaloMetType2"))
)


process.cmsTopTagPFJetsCHS = cms.EDProducer("CATopJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    algorithm = cms.int32(1),
    centralEtaCut = cms.double(2.5),
    debugLevel = cms.untracked.int32(0),
    deltarBins = cms.vdouble(0.19, 0.19, 0.19),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    etFrac = cms.double(0.7),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('caTopSubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nCellBins = cms.vdouble(1.9, 1.9, 1.9),
    nSigmaPU = cms.double(1.0),
    ptFracBins = cms.vdouble(0.05, 0.05, 0.05),
    rBins = cms.vdouble(0.8, 0.8, 0.8),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("ak8PFJetsCHSConstituents","constituents"),
    srcPVs = cms.InputTag(""),
    sumEtBins = cms.vdouble(0, 1600, 2600),
    sumEtEtaCut = cms.double(3.0),
    tagAlgo = cms.int32(0),
    useAdjacency = cms.int32(2),
    useDeterministicSeed = cms.bool(True),
    useMaxTower = cms.bool(False),
    verbose = cms.bool(False),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.cmsTopTagPFJetsCHSMassAK8 = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("cmsTopTagPFJetsCHS"),
    src = cms.InputTag("ak8PFJetsCHS"),
    value = cms.string('mass')
)


process.cmsTopTagPFJetsCHSMassCA8 = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    distMax = cms.double(0.8),
    matched = cms.InputTag("cmsTopTagPFJetsCHS"),
    src = cms.InputTag("ca8PFJetsCHS"),
    value = cms.string('mass')
)


process.combinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('combinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)


process.combinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('combinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)


process.combinedMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('combinedMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.combinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('combinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexTagInfos"))
)


process.combinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('combinedSecondaryVertexSoftLeptonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.combinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('combinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexTagInfos"))
)


process.corrCaloMetType1 = cms.EDProducer("CaloJetMETcorrInputProducer",
    jetCorrEtaMax = cms.double(9.9),
    jetCorrLabel = cms.InputTag("ak4CaloL2L3Corrector"),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.9),
    src = cms.InputTag("ak4CaloJets"),
    srcMET = cms.InputTag("caloMetM"),
    type1JetPtThreshold = cms.double(20.0)
)


process.corrCaloMetType2 = cms.EDProducer("Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(cms.InputTag("corrCaloMetType1","type2"), cms.InputTag("muCaloMetCorr")),
    type2CorrFormula = cms.string('A + B*TMath::Exp(-C*x)'),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)
    )
)


process.corrPfMetType1 = cms.EDProducer("PFJetMETcorrInputProducer",
    jetCorrEtaMax = cms.double(9.9),
    jetCorrLabel = cms.InputTag("ak4PFL1FastL2L3Corrector"),
    offsetCorrLabel = cms.InputTag("ak4PFL1FastjetCorrector"),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.9),
    skipMuonSelection = cms.string('isGlobalMuon | isStandAloneMuon'),
    skipMuons = cms.bool(True),
    src = cms.InputTag("ak4PFJets"),
    type1JetPtThreshold = cms.double(10.0)
)


process.corrPfMetType2 = cms.EDProducer("Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(cms.InputTag("corrPfMetType1","type2"), cms.InputTag("corrPfMetType1","offset"), cms.InputTag("pfCandMETcorr")),
    type2CorrFormula = cms.string('A'),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.4)
    )
)


process.doubleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('doubleVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredTagInfos"))
)


process.elPFIsoDepositChargedAllPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositChargedAllPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedElectronsPFBRECO"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositChargedPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositChargedPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedElectronsPFBRECO"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositGammaPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        MissHitSCMatch_Veto = cms.bool(True),
        SCMatch_Veto = cms.bool(False),
        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositGammaPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        MissHitSCMatch_Veto = cms.bool(True),
        SCMatch_Veto = cms.bool(False),
        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedElectronsPFBRECO"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositNeutralPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllNeutralHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositNeutralPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllNeutralHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedElectronsPFBRECO"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositPUPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfPileUpAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedGsfElectrons"),
    trackType = cms.string('candidate')
)


process.elPFIsoDepositPUPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfPileUpAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedElectronsPFBRECO"),
    trackType = cms.string('candidate')
)


process.elPFIsoValueCharged03NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositCharged"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged03NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged03NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositCharged"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged04NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositCharged"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged04NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged04NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositCharged"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueCharged04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll03NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAll"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll03NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll03NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAll"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll04NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAll"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll04NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll04NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAll"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueChargedAll04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma03NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGamma"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma03NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma03NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGamma"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma04NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGamma"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma04NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma04NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGamma"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueGamma04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral03NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutral"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral03NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral03NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutral"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral04NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutral"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral04NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral04NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutral"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValueNeutral04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU03NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPU"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU03NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU03NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPU"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU04NoPFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPU"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU04NoPFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU04NoPFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPU"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.elPFIsoValuePU04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("elPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
        weight = cms.string('1')
    ))
)


process.electronMatch = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(True),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(0.5),
    maxDeltaR = cms.double(0.5),
    mcPdgId = cms.vint32(11),
    mcStatus = cms.vint32(1),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("gedGsfElectrons")
)


process.fixedGridRhoAll = cms.EDProducer("FixedGridRhoProducer",
    EtaRegion = cms.string('All'),
    pfCandidatesTag = cms.InputTag("particleFlow")
)


process.fixedGridRhoFastjetAll = cms.EDProducer("FixedGridRhoProducerFastjet",
    gridSpacing = cms.double(0.55),
    maxRapidity = cms.double(5.0),
    pfCandidatesTag = cms.InputTag("particleFlow")
)


process.fixedGridRhoFastjetCentralChargedPileUp = cms.EDProducer("FixedGridRhoProducerFastjet",
    gridSpacing = cms.double(0.55),
    maxRapidity = cms.double(2.5),
    pfCandidatesTag = cms.InputTag("particleFlow"),
    src = cms.InputTag("pfPileUpAllChargedParticles")
)


process.fixedGridRhoFastjetCentralNeutral = cms.EDProducer("FixedGridRhoProducerFastjet",
    gridSpacing = cms.double(0.55),
    maxRapidity = cms.double(2.5),
    pfCandidatesTag = cms.InputTag("particleFlow"),
    src = cms.InputTag("pfAllNeutralHadronsAndPhotons")
)


process.genParticlesForJets = cms.EDProducer("InputGenJetsParticleSelector",
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    excludeResonances = cms.bool(False),
    ignoreParticleIDs = cms.vuint32(1000022, 1000012, 1000014, 1000016, 2000012, 
        2000014, 2000016, 1000039, 5100039, 4000012, 
        4000014, 4000016, 9900012, 9900014, 9900016, 
        39),
    partonicFinalState = cms.bool(False),
    src = cms.InputTag("genParticles"),
    tausAsJets = cms.bool(False)
)


process.genParticlesForJetsNoMuNoNu = cms.EDProducer("InputGenJetsParticleSelector",
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    excludeResonances = cms.bool(False),
    ignoreParticleIDs = cms.vuint32(1000022, 1000012, 1000014, 1000016, 2000012, 
        2000014, 2000016, 1000039, 5100039, 4000012, 
        4000014, 4000016, 9900012, 9900014, 9900016, 
        39, 12, 13, 14, 16),
    partonicFinalState = cms.bool(False),
    src = cms.InputTag("genParticles"),
    tausAsJets = cms.bool(False)
)


process.genParticlesForJetsNoNu = cms.EDProducer("InputGenJetsParticleSelector",
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    excludeResonances = cms.bool(False),
    ignoreParticleIDs = cms.vuint32(1000022, 1000012, 1000014, 1000016, 2000012, 
        2000014, 2000016, 1000039, 5100039, 4000012, 
        4000014, 4000016, 9900012, 9900014, 9900016, 
        39, 12, 14, 16),
    partonicFinalState = cms.bool(False),
    src = cms.InputTag("genParticles"),
    tausAsJets = cms.bool(False)
)


process.ghostTrackBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('ghostTrackComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("ghostTrackVertexTagInfos"))
)


process.ghostTrackVertexTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("secondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(False),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(3.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(1),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('gtvr'),
        fitType = cms.string('RefitGhostTrackWithVertices'),
        maxFitChi2 = cms.double(10.0),
        mergeThreshold = cms.double(3.0),
        primcut = cms.double(2.0),
        seccut = cms.double(4.0)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.gk5GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.gk5GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.gk5GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.gk5PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.gk7PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.7),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.gk8GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.gk8GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.gk8GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('GeneralizedKt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.hepTopTagInfos = cms.EDProducer("CATopJetTagger",
    MinMassMax = cms.double(200.0),
    MinMassMin = cms.double(0.0),
    TopMass = cms.double(173),
    TopMassMax = cms.double(250.0),
    TopMassMin = cms.double(0.0),
    WMass = cms.double(80.4),
    WMassMax = cms.double(200.0),
    WMassMin = cms.double(0.0),
    src = cms.InputTag("hepTopTagPFJetsCHS"),
    verbose = cms.bool(False)
)


process.hepTopTagPFJetsCHS = cms.EDProducer("CATopJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    algorithm = cms.int32(1),
    centralEtaCut = cms.double(2.5),
    debugLevel = cms.untracked.int32(0),
    deltarBins = cms.vdouble(0.19, 0.19, 0.19),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    etFrac = cms.double(0.7),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('caTopSubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    maxSubjetMass = cms.double(30.0),
    minSeed = cms.uint32(14327),
    muCut = cms.double(0.8),
    nCellBins = cms.vdouble(1.9, 1.9, 1.9),
    nSigmaPU = cms.double(1.0),
    ptFracBins = cms.vdouble(0.05, 0.05, 0.05),
    rBins = cms.vdouble(0.8, 0.8, 0.8),
    rParam = cms.double(1.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("ak8PFJetsCHSConstituents","constituents"),
    srcPVs = cms.InputTag(""),
    sumEtBins = cms.vdouble(0, 1600, 2600),
    sumEtEtaCut = cms.double(3.0),
    tagAlgo = cms.int32(2),
    useAdjacency = cms.int32(2),
    useDeterministicSeed = cms.bool(True),
    useMaxTower = cms.bool(False),
    useSubjetMass = cms.bool(False),
    verbose = cms.bool(False),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.ic5CaloL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector")
)


process.ic5CaloL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL1OffsetCorrector", "ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector", "ic5CaloL6SLBCorrector")
)


process.ic5CaloL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL1FastjetCorrector", "ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector", "ic5CaloResidualCorrector")
)


process.ic5CaloL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ic5CaloL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL1OffsetCorrector", "ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector")
)


process.ic5CaloL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL1OffsetCorrector", "ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector", "ic5CaloResidualCorrector")
)


process.ic5CaloL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ic5CaloL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector")
)


process.ic5CaloL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector", "ic5CaloL6SLBCorrector")
)


process.ic5CaloL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5CaloL2RelativeCorrector", "ic5CaloL3AbsoluteCorrector", "ic5CaloResidualCorrector")
)


process.ic5CaloL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L2Relative')
)


process.ic5CaloL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L3Absolute')
)


process.ic5CaloL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ic5CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ic5CaloJetsSoftMuonTagInfos")
)


process.ic5CaloResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.ic5PFL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector")
)


process.ic5PFL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector", "ic5PFL6SLBCorrector")
)


process.ic5PFL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5PFL1FastjetCorrector", "ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector", "ic5PFResidualCorrector")
)


process.ic5PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ic5PFL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5PFL1OffsetCorrector", "ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector")
)


process.ic5PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5PFL1OffsetCorrector", "ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector", "ic5PFResidualCorrector")
)


process.ic5PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.ic5PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector")
)


process.ic5PFL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector", "ic5PFL6SLBCorrector")
)


process.ic5PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ic5PFL2RelativeCorrector", "ic5PFL3AbsoluteCorrector", "ic5PFResidualCorrector")
)


process.ic5PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L2Relative')
)


process.ic5PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L3Absolute')
)


process.ic5PFL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ic5PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ic5PFJetsSoftMuonTagInfos")
)


process.ic5PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.impactParameterMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('impactParameterMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.impactParameterTagInfos = cms.EDProducer("TrackIPProducer",
    computeGhostTrack = cms.bool(True),
    computeProbabilities = cms.bool(True),
    ghostTrackPriorDeltaR = cms.double(0.03),
    jetDirectionUsingGhostTrack = cms.bool(False),
    jetDirectionUsingTracks = cms.bool(False),
    jetTracks = cms.InputTag("jetTracksAssociatorAtVertex"),
    maximumChiSquared = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumNumberOfPixelHits = cms.int32(2),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    useTrackQuality = cms.bool(False)
)


process.inclusiveSecondaryVertexFinderFilteredNegativeTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("bToCharmDecayVertexMerged"),
    extSVDeltaRToJet = cms.double(-0.4),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(-2.0),
        distSig2dMin = cms.double(-99999.9),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(-0.01),
        distVal2dMin = cms.double(-2.5),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(-0.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.inclusiveSecondaryVertexFinderFilteredTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("bToCharmDecayVertexMerged"),
    extSVDeltaRToJet = cms.double(0.4),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(2.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.inclusiveSecondaryVertexFinderNegativeTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveSecondaryVertices"),
    extSVDeltaRToJet = cms.double(-0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(-2.0),
        distSig2dMin = cms.double(-99999.9),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(-0.01),
        distVal2dMin = cms.double(-2.5),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(-0.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.inclusiveSecondaryVertexFinderTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveSecondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(2.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.isoDeposits = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag(""),
    trackType = cms.string('candidate')
)


process.iterativeCone5GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('IterativeCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.iterativeCone5GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('IterativeCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.iterativeCone5GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('IterativeCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.iterativeCone5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('IterativeCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.iterativeCone5PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('IterativeCone'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.iterativeCone7HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('IterativeCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.7),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.jetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('jetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.jetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('jetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.jetTracksAssociatorAtVertex = cms.EDProducer("JetTracksAssociatorAtVertex",
    coneSize = cms.double(0.4),
    jets = cms.InputTag("ak4PFJets"),
    pvSrc = cms.InputTag("offlinePrimaryVertices"),
    tracks = cms.InputTag("generalTracks"),
    useAssigned = cms.bool(False)
)


process.jhuTopTagPFJetsCHS = cms.EDProducer("CATopJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    algorithm = cms.int32(1),
    centralEtaCut = cms.double(2.5),
    cosThetaWMax = cms.double(0.7),
    debugLevel = cms.untracked.int32(0),
    deltaRCut = cms.double(0.19),
    deltarBins = cms.vdouble(0.19, 0.19, 0.19),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    etFrac = cms.double(0.7),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jetCollInstanceName = cms.string('caTopSubJets'),
    jetPtMin = cms.double(100.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nCellBins = cms.vdouble(1.9, 1.9, 1.9),
    nSigmaPU = cms.double(1.0),
    ptFrac = cms.double(0.05),
    ptFracBins = cms.vdouble(0.05, 0.05, 0.05),
    rBins = cms.vdouble(0.8, 0.8, 0.8),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfNoPileUpJME"),
    srcPVs = cms.InputTag(""),
    sumEtBins = cms.vdouble(0, 1600, 2600),
    sumEtEtaCut = cms.double(3.0),
    tagAlgo = cms.int32(0),
    useAdjacency = cms.int32(2),
    useDeterministicSeed = cms.bool(True),
    useMaxTower = cms.bool(False),
    verbose = cms.bool(False),
    voronoiRfact = cms.double(-0.9),
    writeCompound = cms.bool(True)
)


process.kt4CaloL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector")
)


process.kt4CaloL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL1OffsetCorrector", "kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector", "kt4CaloL6SLBCorrector")
)


process.kt4CaloL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL1FastjetCorrector", "kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector", "kt4CaloResidualCorrector")
)


process.kt4CaloL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.kt4CaloL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL1OffsetCorrector", "kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector")
)


process.kt4CaloL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL1OffsetCorrector", "kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector", "kt4CaloResidualCorrector")
)


process.kt4CaloL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.kt4CaloL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector")
)


process.kt4CaloL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector", "kt4CaloL6SLBCorrector")
)


process.kt4CaloL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4CaloL2RelativeCorrector", "kt4CaloL3AbsoluteCorrector", "kt4CaloResidualCorrector")
)


process.kt4CaloL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L2Relative')
)


process.kt4CaloL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L3Absolute')
)


process.kt4CaloL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt4CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt4CaloJetsSoftMuonTagInfos")
)


process.kt4CaloResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.kt4GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt4GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt4GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt4HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt4PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.4),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.kt4PFL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector")
)


process.kt4PFL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector", "kt4PFL6SLBCorrector")
)


process.kt4PFL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4PFL1FastjetCorrector", "kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector", "kt4PFResidualCorrector")
)


process.kt4PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.kt4PFL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4PFL1OffsetCorrector", "kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector")
)


process.kt4PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4PFL1OffsetCorrector", "kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector", "kt4PFResidualCorrector")
)


process.kt4PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.kt4PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector")
)


process.kt4PFL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector", "kt4PFL6SLBCorrector")
)


process.kt4PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt4PFL2RelativeCorrector", "kt4PFL3AbsoluteCorrector", "kt4PFResidualCorrector")
)


process.kt4PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L2Relative')
)


process.kt4PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L3Absolute')
)


process.kt4PFL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt4PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt4PFJetsSoftMuonTagInfos")
)


process.kt4PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.kt6CaloL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4CaloL1FastjetCorrector", "kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector")
)


process.kt6CaloL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL1OffsetCorrector", "kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector", "kt6CaloL6SLBCorrector")
)


process.kt6CaloL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL1FastjetCorrector", "kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector", "kt6CaloResidualCorrector")
)


process.kt6CaloL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.kt6CaloL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL1OffsetCorrector", "kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector")
)


process.kt6CaloL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL1OffsetCorrector", "kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector", "kt6CaloResidualCorrector")
)


process.kt6CaloL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.kt6CaloL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector")
)


process.kt6CaloL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector", "kt6CaloL6SLBCorrector")
)


process.kt6CaloL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6CaloL2RelativeCorrector", "kt6CaloL3AbsoluteCorrector", "kt6CaloResidualCorrector")
)


process.kt6CaloL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L2Relative')
)


process.kt6CaloL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L3Absolute')
)


process.kt6CaloL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt6CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt6CaloJetsSoftMuonTagInfos")
)


process.kt6CaloResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.kt6GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt6GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt6GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt6HiGenJets = cms.EDProducer("SubEventGenJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.kt6PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(True),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(0.9)
)


process.kt6PFJetsCentralChargedPileUp = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(3.1),
    Rho_EtaMax = cms.double(2.5),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(True),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfPileUpAllChargedParticles"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(0.9)
)


process.kt6PFJetsCentralNeutral = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(3.1),
    Rho_EtaMax = cms.double(2.5),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(True),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfAllNeutralHadronsAndPhotons"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(0.9)
)


process.kt6PFJetsCentralNeutralTight = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(3.1),
    Rho_EtaMax = cms.double(2.5),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(True),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(True),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(1.0),
    jetAlgorithm = cms.string('Kt'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.6),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("pfAllNeutralHadronsAndPhotons"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(0.9)
)


process.kt6PFL1FastL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector")
)


process.kt6PFL1FastL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("ak4PFL1FastjetCorrector", "kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector", "kt6PFL6SLBCorrector")
)


process.kt6PFL1FastL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6PFL1FastjetCorrector", "kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector", "kt6PFResidualCorrector")
)


process.kt6PFL1FastjetCorrector = cms.EDProducer("L1FastjetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.kt6PFL1L2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6PFL1OffsetCorrector", "kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector")
)


process.kt6PFL1L2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6PFL1OffsetCorrector", "kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector", "kt6PFResidualCorrector")
)


process.kt6PFL1OffsetCorrector = cms.EDProducer("L1OffsetCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)


process.kt6PFL2L3Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector")
)


process.kt6PFL2L3L6Corrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector", "kt6PFL6SLBCorrector")
)


process.kt6PFL2L3ResidualCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("kt6PFL2RelativeCorrector", "kt6PFL3AbsoluteCorrector", "kt6PFResidualCorrector")
)


process.kt6PFL2RelativeCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L2Relative')
)


process.kt6PFL3AbsoluteCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L3Absolute')
)


process.kt6PFL6SLBCorrector = cms.EDProducer("L6SLBCorrectorProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt6PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt6PFJetsSoftMuonTagInfos")
)


process.kt6PFResidualCorrector = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.muCaloMetCorr = cms.EDProducer("MuonMETcorrInputProducer",
    src = cms.InputTag("muons"),
    srcMuonCorrections = cms.InputTag("muonMETValueMapProducer","muCorrData")
)


process.muPFIsoDepositChargedAllPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("muons"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositChargedAllPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedMuonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositChargedPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("muons"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositChargedPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedMuonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositGammaPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("muons"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositGammaPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedMuonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositNeutralPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllNeutralHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("muons"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositNeutralPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllNeutralHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedMuonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositPUPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfPileUpAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("muons"),
    trackType = cms.string('candidate')
)


process.muPFIsoDepositPUPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(1e-05),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfPileUpAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedMuonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.muPFIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositCharged"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueCharged03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueCharged03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositCharged"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueCharged04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueCharged04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAll"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueChargedAll03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueChargedAll03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAll"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueChargedAll04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueChargedAll04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGamma03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGamma03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGamma04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGamma04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGammaHighThreshold03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGammaHighThreshold03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGammaHighThreshold04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueGammaHighThreshold04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutral03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutral03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutral04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutral04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutralHighThreshold03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutralHighThreshold03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutralHighThreshold04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValueNeutralHighThreshold04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPU"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValuePU03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValuePU03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPU"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValuePU04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFIsoValuePU04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositCharged"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueCharged03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueCharged03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositCharged"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueCharged04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueCharged04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAll"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueChargedAll03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueChargedAll03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAll"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueChargedAll04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueChargedAll04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGamma03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGamma03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGamma04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGamma04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGammaHighThreshold03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGammaHighThreshold03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGammaHighThreshold04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueGammaHighThreshold04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutral03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutral03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutral04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutral04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutralHighThreshold03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutralHighThreshold03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutralHighThreshold04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValueNeutralHighThreshold04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPU"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValuePU03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValuePU03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPU"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValuePU04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFMeanDRIsoValuePU04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('meanDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositCharged"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueCharged03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueCharged03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositCharged"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueCharged04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueCharged04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAll"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueChargedAll03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueChargedAll03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAll"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueChargedAll04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueChargedAll04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring('0.0001', 
            'Threshold(0.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGamma03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGamma03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGamma04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGamma04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGammaHighThreshold03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGammaHighThreshold03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGamma"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGammaHighThreshold04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueGammaHighThreshold04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutral03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutral03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutral04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutral04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutralHighThreshold03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutralHighThreshold03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutral"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutralHighThreshold04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValueNeutralHighThreshold04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(1.0)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPU"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValuePU03PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValuePU03PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPU"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValuePU04PAT = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPAT"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muPFSumDRIsoValuePU04PFBRECO = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.4),
        mode = cms.string('sumDR'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("muPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring('0.01', 
            'Threshold(0.5)'),
        weight = cms.string('1')
    ))
)


process.muonMatch = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(True),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(0.5),
    maxDeltaR = cms.double(0.5),
    mcPdgId = cms.vint32(13),
    mcStatus = cms.vint32(1),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("muons")
)


process.negativeCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"))
)


process.negativeCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"))
)


process.negativeCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeCombinedMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderNegativeTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.negativeCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexNegativeTagInfos"))
)


process.negativeCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexNegativeTagInfos"))
)


process.negativeOnlyJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeOnlyJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.negativeOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeOnlyJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.negativeSimpleInclusiveSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredNegativeTagInfos"))
)


process.negativeSimpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredNegativeTagInfos"))
)


process.negativeSimpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)


process.negativeSimpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexNegativeTagInfos"))
)


process.negativeSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.negativeSoftPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.negativeSoftPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.negativeSoftPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.negativeSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.negativeSoftPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.negativeSoftPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.negativeSoftPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.negativeTrackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeTrackCounting3D2ndComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.negativeTrackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeTrackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.particleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
    src = cms.InputTag("particleFlow")
)


process.patElectrons = cms.EDProducer("PATElectronProducer",
    addEfficiencies = cms.bool(False),
    addElectronID = cms.bool(True),
    addGenMatch = cms.bool(True),
    addPFClusterIso = cms.bool(False),
    addResolutions = cms.bool(False),
    beamLineSrc = cms.InputTag("offlineBeamSpot"),
    efficiencies = cms.PSet(

    ),
    electronIDSources = cms.PSet(
        eidLoose = cms.InputTag("eidLoose"),
        eidRobustHighEnergy = cms.InputTag("eidRobustHighEnergy"),
        eidRobustLoose = cms.InputTag("eidRobustLoose"),
        eidRobustTight = cms.InputTag("eidRobustTight"),
        eidTight = cms.InputTag("eidTight")
    ),
    electronSource = cms.InputTag("gedGsfElectrons"),
    embedBasicClusters = cms.bool(True),
    embedGenMatch = cms.bool(True),
    embedGsfElectronCore = cms.bool(True),
    embedGsfTrack = cms.bool(True),
    embedHighLevelSelection = cms.bool(True),
    embedPFCandidate = cms.bool(True),
    embedPflowBasicClusters = cms.bool(True),
    embedPflowPreshowerClusters = cms.bool(True),
    embedPflowSuperCluster = cms.bool(True),
    embedPreshowerClusters = cms.bool(True),
    embedRecHits = cms.bool(True),
    embedSeedCluster = cms.bool(True),
    embedSuperCluster = cms.bool(True),
    embedTrack = cms.bool(True),
    genParticleMatch = cms.InputTag("electronMatch"),
    isoDeposits = cms.PSet(
        pfChargedAll = cms.InputTag("elPFIsoDepositChargedAllPAT"),
        pfChargedHadrons = cms.InputTag("elPFIsoDepositChargedPAT"),
        pfNeutralHadrons = cms.InputTag("elPFIsoDepositNeutralPAT"),
        pfPUChargedHadrons = cms.InputTag("elPFIsoDepositPUPAT"),
        pfPhotons = cms.InputTag("elPFIsoDepositGammaPAT")
    ),
    isolationValues = cms.PSet(
        pfChargedAll = cms.InputTag("elPFIsoValueChargedAll04PFIdPAT"),
        pfChargedHadrons = cms.InputTag("elPFIsoValueCharged04PFIdPAT"),
        pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral04PFIdPAT"),
        pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU04PFIdPAT"),
        pfPhotons = cms.InputTag("elPFIsoValueGamma04PFIdPAT")
    ),
    isolationValuesNoPFId = cms.PSet(
        pfChargedAll = cms.InputTag("elPFIsoValueChargedAll04NoPFIdPAT"),
        pfChargedHadrons = cms.InputTag("elPFIsoValueCharged04NoPFIdPAT"),
        pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral04NoPFIdPAT"),
        pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU04NoPFIdPAT"),
        pfPhotons = cms.InputTag("elPFIsoValueGamma04NoPFIdPAT")
    ),
    pfCandidateMap = cms.InputTag("particleFlow","electrons"),
    pfElectronSource = cms.InputTag("particleFlow"),
    pvSrc = cms.InputTag("offlinePrimaryVertices"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    resolutions = cms.PSet(

    ),
    useParticleFlow = cms.bool(False),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    ),
    userIsolation = cms.PSet(

    )
)


process.patJetCharge = cms.EDProducer("JetChargeProducer",
    exp = cms.double(1.0),
    src = cms.InputTag("jetTracksAssociatorAtVertex"),
    var = cms.string('Pt')
)


process.patJetCorrFactors = cms.EDProducer("JetCorrFactorsProducer",
    emf = cms.bool(False),
    extraJPTOffset = cms.string('L1FastJet'),
    flavorType = cms.string('J'),
    levels = cms.vstring('L1FastJet', 
        'L2Relative', 
        'L3Absolute'),
    payload = cms.string('AK5PF'),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    src = cms.InputTag("ak4PFJets"),
    useNPV = cms.bool(True),
    useRho = cms.bool(True)
)


process.patJetCorrFactorsAK4PFCHS = cms.EDProducer("JetCorrFactorsProducer",
    emf = cms.bool(False),
    extraJPTOffset = cms.string('L1FastJet'),
    flavorType = cms.string('J'),
    levels = cms.vstring('L1FastJet', 
        'L2Relative', 
        'L3Absolute'),
    payload = cms.string('AK5PFchs'),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    src = cms.InputTag("ak4PFJetsCHS"),
    useNPV = cms.bool(True),
    useRho = cms.bool(True)
)


process.patJetCorrFactorsAK8PFCHS = cms.EDProducer("JetCorrFactorsProducer",
    emf = cms.bool(False),
    extraJPTOffset = cms.string('L1FastJet'),
    flavorType = cms.string('J'),
    levels = cms.vstring('L1FastJet', 
        'L2Relative', 
        'L3Absolute'),
    payload = cms.string('AK7PFchs'),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    src = cms.InputTag("ak8PFJetsCHS"),
    useNPV = cms.bool(True),
    useRho = cms.bool(True)
)


process.patJetCorrFactorsCA8PFCHS = cms.EDProducer("JetCorrFactorsProducer",
    emf = cms.bool(False),
    extraJPTOffset = cms.string('L1FastJet'),
    flavorType = cms.string('J'),
    levels = cms.vstring('L1FastJet', 
        'L2Relative', 
        'L3Absolute'),
    payload = cms.string('AK7PFchs'),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    src = cms.InputTag("ca8PFJetsCHS"),
    useNPV = cms.bool(True),
    useRho = cms.bool(True)
)


process.patJetFlavourAssociation = cms.EDProducer("JetFlavourClustering",
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(True),
    jetAlgorithm = cms.string('AntiKt'),
    jets = cms.InputTag("ak4PFJets"),
    partons = cms.InputTag("patJetPartons","partons"),
    rParam = cms.double(0.4)
)


process.patJetFlavourAssociationAK4PFCHS = cms.EDProducer("JetFlavourClustering",
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(True),
    jetAlgorithm = cms.string('AntiKt'),
    jets = cms.InputTag("ak4PFJetsCHS"),
    partons = cms.InputTag("patJetPartons","partons"),
    rParam = cms.double(0.4)
)


process.patJetFlavourAssociationAK8PFCHS = cms.EDProducer("JetFlavourClustering",
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(True),
    jetAlgorithm = cms.string('AntiKt'),
    jets = cms.InputTag("ak8PFJetsCHS"),
    partons = cms.InputTag("patJetPartons","partons"),
    rParam = cms.double(0.8)
)


process.patJetFlavourAssociationCA8PFCHS = cms.EDProducer("JetFlavourClustering",
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(True),
    jetAlgorithm = cms.string('CambridgeAachen'),
    jets = cms.InputTag("ca8PFJetsCHS"),
    partons = cms.InputTag("patJetPartons","partons"),
    rParam = cms.double(0.8)
)


process.patJetFlavourAssociationLegacy = cms.EDProducer("JetFlavourIdentifier",
    physicsDefinition = cms.bool(False),
    srcByReference = cms.InputTag("patJetPartonAssociationLegacy")
)


process.patJetFlavourAssociationLegacyAK4PFCHS = cms.EDProducer("JetFlavourIdentifier",
    physicsDefinition = cms.bool(False),
    srcByReference = cms.InputTag("patJetPartonAssociationLegacyAK4PFCHS")
)


process.patJetFlavourAssociationLegacyAK8PFCHS = cms.EDProducer("JetFlavourIdentifier",
    physicsDefinition = cms.bool(False),
    srcByReference = cms.InputTag("patJetPartonAssociationLegacyAK8PFCHS")
)


process.patJetFlavourAssociationLegacyCA8PFCHS = cms.EDProducer("JetFlavourIdentifier",
    physicsDefinition = cms.bool(False),
    srcByReference = cms.InputTag("patJetPartonAssociationLegacyCA8PFCHS")
)


process.patJetGenJetMatch = cms.EDProducer("GenJetMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = cms.double(0.4),
    mcPdgId = cms.vint32(),
    mcStatus = cms.vint32(),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ak4PFJets")
)


process.patJetGenJetMatchAK4PFCHS = cms.EDProducer("GenJetMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = cms.double(0.4),
    mcPdgId = cms.vint32(),
    mcStatus = cms.vint32(),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ak4PFJetsCHS")
)


process.patJetGenJetMatchAK8PFCHS = cms.EDProducer("GenJetMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = cms.double(0.8),
    mcPdgId = cms.vint32(),
    mcStatus = cms.vint32(),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ak8PFJetsCHS")
)


process.patJetGenJetMatchCA8PFCHS = cms.EDProducer("GenJetMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = cms.double(0.8),
    mcPdgId = cms.vint32(),
    mcStatus = cms.vint32(),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ca8PFJetsCHS")
)


process.patJetPartonAssociationLegacy = cms.EDProducer("JetPartonMatcher",
    coneSizeToAssociate = cms.double(0.3),
    jets = cms.InputTag("ak4PFJets"),
    partons = cms.InputTag("patJetPartonsLegacy")
)


process.patJetPartonAssociationLegacyAK4PFCHS = cms.EDProducer("JetPartonMatcher",
    coneSizeToAssociate = cms.double(0.3),
    jets = cms.InputTag("ak4PFJetsCHS"),
    partons = cms.InputTag("patJetPartonsLegacy")
)


process.patJetPartonAssociationLegacyAK8PFCHS = cms.EDProducer("JetPartonMatcher",
    coneSizeToAssociate = cms.double(0.3),
    jets = cms.InputTag("ak8PFJetsCHS"),
    partons = cms.InputTag("patJetPartonsLegacy")
)


process.patJetPartonAssociationLegacyCA8PFCHS = cms.EDProducer("JetPartonMatcher",
    coneSizeToAssociate = cms.double(0.3),
    jets = cms.InputTag("ca8PFJetsCHS"),
    partons = cms.InputTag("patJetPartonsLegacy")
)


process.patJetPartonMatch = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(3.0),
    maxDeltaR = cms.double(0.4),
    mcPdgId = cms.vint32(1, 2, 3, 4, 5, 
        21),
    mcStatus = cms.vint32(3),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ak4PFJets")
)


process.patJetPartonMatchAK4PFCHS = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(3.0),
    maxDeltaR = cms.double(0.4),
    mcPdgId = cms.vint32(1, 2, 3, 4, 5, 
        21),
    mcStatus = cms.vint32(3),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ak4PFJetsCHS")
)


process.patJetPartonMatchAK8PFCHS = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(3.0),
    maxDeltaR = cms.double(0.4),
    mcPdgId = cms.vint32(1, 2, 3, 4, 5, 
        21),
    mcStatus = cms.vint32(3),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ak8PFJetsCHS")
)


process.patJetPartonMatchCA8PFCHS = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(3.0),
    maxDeltaR = cms.double(0.4),
    mcPdgId = cms.vint32(1, 2, 3, 4, 5, 
        21),
    mcStatus = cms.vint32(3),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("ca8PFJetsCHS")
)


process.patJetPartons = cms.EDProducer("HadronAndPartonSelector",
    particles = cms.InputTag("genParticles"),
    partonMode = cms.string('Auto'),
    src = cms.InputTag("generator")
)


process.patJetPartonsLegacy = cms.EDProducer("PartonSelector",
    src = cms.InputTag("genParticles"),
    withLeptons = cms.bool(False)
)


process.patJets = cms.EDProducer("PATJetProducer",
    JetFlavourInfoSource = cms.InputTag("patJetFlavourAssociation"),
    JetPartonMapSource = cms.InputTag("patJetFlavourAssociationLegacy"),
    addAssociatedTracks = cms.bool(True),
    addBTagInfo = cms.bool(True),
    addDiscriminators = cms.bool(True),
    addEfficiencies = cms.bool(False),
    addGenJetMatch = cms.bool(True),
    addGenPartonMatch = cms.bool(True),
    addJetCharge = cms.bool(True),
    addJetCorrFactors = cms.bool(True),
    addJetFlavourInfo = cms.bool(False),
    addJetID = cms.bool(False),
    addPartonJetMatch = cms.bool(False),
    addResolutions = cms.bool(False),
    addTagInfos = cms.bool(False),
    discriminatorSources = cms.VInputTag(cms.InputTag("jetBProbabilityBJetTags"), cms.InputTag("jetProbabilityBJetTags"), cms.InputTag("trackCountingHighPurBJetTags"), cms.InputTag("trackCountingHighEffBJetTags"), cms.InputTag("simpleSecondaryVertexHighEffBJetTags"), 
        cms.InputTag("simpleSecondaryVertexHighPurBJetTags"), cms.InputTag("combinedSecondaryVertexBJetTags")),
    efficiencies = cms.PSet(

    ),
    embedGenJetMatch = cms.bool(True),
    embedGenPartonMatch = cms.bool(True),
    embedPFCandidates = cms.bool(False),
    genJetMatch = cms.InputTag("patJetGenJetMatch"),
    genPartonMatch = cms.InputTag("patJetPartonMatch"),
    getJetMCFlavour = cms.bool(True),
    jetChargeSource = cms.InputTag("patJetCharge"),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactors")),
    jetIDMap = cms.InputTag("ak4JetID"),
    jetSource = cms.InputTag("ak4PFJets"),
    partonJetSource = cms.InputTag("NOT_IMPLEMENTED"),
    resolutions = cms.PSet(

    ),
    tagInfoSources = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexTagInfos")),
    trackAssociationSource = cms.InputTag("jetTracksAssociatorAtVertex"),
    useLegacyJetMCFlavour = cms.bool(False),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    )
)


process.patJetsAK4PFCHS = cms.EDProducer("PATJetProducer",
    JetFlavourInfoSource = cms.InputTag("patJetFlavourAssociationAK4PFCHS"),
    JetPartonMapSource = cms.InputTag("patJetFlavourAssociationLegacyAK4PFCHS"),
    addAssociatedTracks = cms.bool(False),
    addBTagInfo = cms.bool(False),
    addDiscriminators = cms.bool(True),
    addEfficiencies = cms.bool(False),
    addGenJetMatch = cms.bool(True),
    addGenPartonMatch = cms.bool(True),
    addJetCharge = cms.bool(False),
    addJetCorrFactors = cms.bool(True),
    addJetFlavourInfo = cms.bool(False),
    addJetID = cms.bool(False),
    addPartonJetMatch = cms.bool(False),
    addResolutions = cms.bool(False),
    addTagInfos = cms.bool(False),
    discriminatorSources = cms.VInputTag(cms.InputTag("combinedSecondaryVertexBJetTags"), cms.InputTag("pfJetBProbabilityBJetTags"), cms.InputTag("pfJetProbabilityBJetTags"), cms.InputTag("pfTrackCountingHighPurBJetTags"), cms.InputTag("pfTrackCountingHighEffBJetTags"), 
        cms.InputTag("pfSimpleSecondaryVertexHighEffBJetTags"), cms.InputTag("pfSimpleSecondaryVertexHighPurBJetTags"), cms.InputTag("pfCombinedSecondaryVertexV2BJetTags"), cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"), cms.InputTag("pfCombinedSecondaryVertexSoftLeptonBJetTags"), 
        cms.InputTag("pfCombinedMVABJetTags")),
    efficiencies = cms.PSet(

    ),
    embedGenJetMatch = cms.bool(True),
    embedGenPartonMatch = cms.bool(True),
    embedPFCandidates = cms.bool(False),
    genJetMatch = cms.InputTag("patJetGenJetMatchAK4PFCHS"),
    genPartonMatch = cms.InputTag("patJetPartonMatchAK4PFCHS"),
    getJetMCFlavour = cms.bool(True),
    jetChargeSource = cms.InputTag(""),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsAK4PFCHS")),
    jetIDMap = cms.InputTag("ak4JetID"),
    jetSource = cms.InputTag("ak4PFJetsCHS"),
    partonJetSource = cms.InputTag("NOT_IMPLEMENTED"),
    resolutions = cms.PSet(

    ),
    tagInfoSources = cms.VInputTag(),
    trackAssociationSource = cms.InputTag(""),
    useLegacyJetMCFlavour = cms.bool(False),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("", "pileupJetIdEvaluator:fullDiscriminant", "QGTagger:qgLikelihood")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("", "pileupJetIdEvaluator:cutbasedId", "pileupJetIdEvaluator:fullId")
        )
    )
)


process.patJetsAK8PFCHS = cms.EDProducer("PATJetProducer",
    JetFlavourInfoSource = cms.InputTag("patJetFlavourAssociationAK8PFCHS"),
    JetPartonMapSource = cms.InputTag("patJetFlavourAssociationLegacyAK8PFCHS"),
    addAssociatedTracks = cms.bool(False),
    addBTagInfo = cms.bool(False),
    addDiscriminators = cms.bool(True),
    addEfficiencies = cms.bool(False),
    addGenJetMatch = cms.bool(True),
    addGenPartonMatch = cms.bool(True),
    addJetCharge = cms.bool(False),
    addJetCorrFactors = cms.bool(True),
    addJetFlavourInfo = cms.bool(False),
    addJetID = cms.bool(False),
    addPartonJetMatch = cms.bool(False),
    addResolutions = cms.bool(False),
    addTagInfos = cms.bool(False),
    discriminatorSources = cms.VInputTag(cms.InputTag("combinedSecondaryVertexBJetTags"), cms.InputTag("pfJetBProbabilityBJetTags"), cms.InputTag("pfJetProbabilityBJetTags"), cms.InputTag("pfTrackCountingHighPurBJetTags"), cms.InputTag("pfTrackCountingHighEffBJetTags"), 
        cms.InputTag("pfSimpleSecondaryVertexHighEffBJetTags"), cms.InputTag("pfSimpleSecondaryVertexHighPurBJetTags"), cms.InputTag("pfCombinedSecondaryVertexV2BJetTags"), cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"), cms.InputTag("pfCombinedSecondaryVertexSoftLeptonBJetTags"), 
        cms.InputTag("pfCombinedMVABJetTags")),
    efficiencies = cms.PSet(

    ),
    embedGenJetMatch = cms.bool(True),
    embedGenPartonMatch = cms.bool(True),
    embedPFCandidates = cms.bool(False),
    genJetMatch = cms.InputTag("patJetGenJetMatchAK8PFCHS"),
    genPartonMatch = cms.InputTag("patJetPartonMatchAK8PFCHS"),
    getJetMCFlavour = cms.bool(True),
    jetChargeSource = cms.InputTag(""),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsAK8PFCHS")),
    jetIDMap = cms.InputTag("ak4JetID"),
    jetSource = cms.InputTag("ak8PFJetsCHS"),
    partonJetSource = cms.InputTag("NOT_IMPLEMENTED"),
    resolutions = cms.PSet(

    ),
    tagInfoSources = cms.VInputTag(),
    trackAssociationSource = cms.InputTag(""),
    useLegacyJetMCFlavour = cms.bool(False),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("", "NjettinessAK8:tau1", "NjettinessAK8:tau2", "NjettinessAK8:tau3", "QJetsAdderAK8:QjetsVolatility", 
                "ak8PFJetsCHSPrunedMass", "ak8PFJetsCHSSoftDropMass", "ak8PFJetsCHSTrimmedMass", "ak8PFJetsCHSFilteredMass", "cmsTopTagPFJetsCHSMassAK8")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    )
)


process.patJetsCA8PFCHS = cms.EDProducer("PATJetProducer",
    JetFlavourInfoSource = cms.InputTag("patJetFlavourAssociationCA8PFCHS"),
    JetPartonMapSource = cms.InputTag("patJetFlavourAssociationLegacyCA8PFCHS"),
    addAssociatedTracks = cms.bool(False),
    addBTagInfo = cms.bool(False),
    addDiscriminators = cms.bool(True),
    addEfficiencies = cms.bool(False),
    addGenJetMatch = cms.bool(True),
    addGenPartonMatch = cms.bool(True),
    addJetCharge = cms.bool(False),
    addJetCorrFactors = cms.bool(True),
    addJetFlavourInfo = cms.bool(False),
    addJetID = cms.bool(False),
    addPartonJetMatch = cms.bool(False),
    addResolutions = cms.bool(False),
    addTagInfos = cms.bool(False),
    discriminatorSources = cms.VInputTag(cms.InputTag("combinedSecondaryVertexBJetTags"), cms.InputTag("pfJetBProbabilityBJetTags"), cms.InputTag("pfJetProbabilityBJetTags"), cms.InputTag("pfTrackCountingHighPurBJetTags"), cms.InputTag("pfTrackCountingHighEffBJetTags"), 
        cms.InputTag("pfSimpleSecondaryVertexHighEffBJetTags"), cms.InputTag("pfSimpleSecondaryVertexHighPurBJetTags"), cms.InputTag("pfCombinedSecondaryVertexV2BJetTags"), cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"), cms.InputTag("pfCombinedSecondaryVertexSoftLeptonBJetTags"), 
        cms.InputTag("pfCombinedMVABJetTags")),
    efficiencies = cms.PSet(

    ),
    embedGenJetMatch = cms.bool(True),
    embedGenPartonMatch = cms.bool(True),
    embedPFCandidates = cms.bool(False),
    genJetMatch = cms.InputTag("patJetGenJetMatchCA8PFCHS"),
    genPartonMatch = cms.InputTag("patJetPartonMatchCA8PFCHS"),
    getJetMCFlavour = cms.bool(True),
    jetChargeSource = cms.InputTag(""),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("patJetCorrFactorsCA8PFCHS")),
    jetIDMap = cms.InputTag("ak4JetID"),
    jetSource = cms.InputTag("ca8PFJetsCHS"),
    partonJetSource = cms.InputTag("NOT_IMPLEMENTED"),
    resolutions = cms.PSet(

    ),
    tagInfoSources = cms.VInputTag(),
    trackAssociationSource = cms.InputTag(""),
    useLegacyJetMCFlavour = cms.bool(False),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("", "NjettinessCA8:tau1", "NjettinessCA8:tau2", "NjettinessCA8:tau3", "QJetsAdderCA8:QjetsVolatility", 
                "ca8PFJetsCHSPrunedMass", "ca8PFJetsCHSSoftDropMass", "ca8PFJetsCHSTrimmedMass", "ca8PFJetsCHSFilteredMass", "cmsTopTagPFJetsCHSMassCA8")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    )
)


process.patMETs = cms.EDProducer("PATMETProducer",
    addEfficiencies = cms.bool(False),
    addGenMET = cms.bool(True),
    addMuonCorrections = cms.bool(False),
    addResolutions = cms.bool(False),
    computeMETSignificance = cms.bool(False),
    efficiencies = cms.PSet(

    ),
    genMETSource = cms.InputTag("genMetTrue"),
    metSource = cms.InputTag("AK5PFType1CorMet"),
    muonSource = cms.InputTag("muons"),
    parameters = cms.PSet(
        dRMatch = cms.double(0.4),
        jetThreshold = cms.double(20),
        jeta = cms.vdouble(0.5, 1.1, 1.7, 2.3),
        jpar = cms.vdouble(1.41, 1.29, 1.41, 1.4, 2.53),
        phiResFile = cms.string('Spring10_PhiResolution_AK5PF.txt'),
        pjpar = cms.vdouble(0.0, 0.674),
        ptResFile = cms.string('Spring10_PtResolution_AK5PF.txt')
    ),
    resolutions = cms.PSet(

    ),
    srcJets = cms.InputTag("selectedPatJets"),
    srcLeptons = cms.VInputTag("selectedPatElectrons", "selectedPatMuons", "selectedPatPhotons"),
    srcPFCands = cms.InputTag("packedPFCandidates"),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    )
)


process.patMuons = cms.EDProducer("PATMuonProducer",
    addEfficiencies = cms.bool(False),
    addGenMatch = cms.bool(True),
    addResolutions = cms.bool(False),
    beamLineSrc = cms.InputTag("offlineBeamSpot"),
    caloMETMuonCorrs = cms.InputTag("muonMETValueMapProducer","muCorrData"),
    efficiencies = cms.PSet(

    ),
    embedCaloMETMuonCorrs = cms.bool(True),
    embedCombinedMuon = cms.bool(True),
    embedDytMuon = cms.bool(True),
    embedGenMatch = cms.bool(True),
    embedHighLevelSelection = cms.bool(True),
    embedMuonBestTrack = cms.bool(True),
    embedPFCandidate = cms.bool(True),
    embedPfEcalEnergy = cms.bool(True),
    embedPickyMuon = cms.bool(True),
    embedStandAloneMuon = cms.bool(True),
    embedTcMETMuonCorrs = cms.bool(False),
    embedTpfmsMuon = cms.bool(True),
    embedTrack = cms.bool(False),
    embedTunePMuonBestTrack = cms.bool(True),
    forceBestTrackEmbedding = cms.bool(False),
    genParticleMatch = cms.InputTag("muonMatch"),
    isoDeposits = cms.PSet(
        pfChargedAll = cms.InputTag("muPFIsoDepositChargedAllPAT"),
        pfChargedHadrons = cms.InputTag("muPFIsoDepositChargedPAT"),
        pfNeutralHadrons = cms.InputTag("muPFIsoDepositNeutralPAT"),
        pfPUChargedHadrons = cms.InputTag("muPFIsoDepositPUPAT"),
        pfPhotons = cms.InputTag("muPFIsoDepositGammaPAT")
    ),
    isolationValues = cms.PSet(
        pfChargedAll = cms.InputTag("muPFIsoValueChargedAll04PAT"),
        pfChargedHadrons = cms.InputTag("muPFIsoValueCharged04PAT"),
        pfNeutralHadrons = cms.InputTag("muPFIsoValueNeutral04PAT"),
        pfPUChargedHadrons = cms.InputTag("muPFIsoValuePU04PAT"),
        pfPhotons = cms.InputTag("muPFIsoValueGamma04PAT")
    ),
    muonSource = cms.InputTag("muons"),
    pfMuonSource = cms.InputTag("particleFlow"),
    pvSrc = cms.InputTag("offlinePrimaryVertices"),
    resolutions = cms.PSet(

    ),
    tcMETMuonCorrs = cms.InputTag("muonTCMETValueMapProducer","muCorrData"),
    useParticleFlow = cms.bool(False),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    ),
    userIsolation = cms.PSet(

    )
)


process.patPhotons = cms.EDProducer("PATPhotonProducer",
    addEfficiencies = cms.bool(False),
    addGenMatch = cms.bool(True),
    addPFClusterIso = cms.bool(False),
    addPhotonID = cms.bool(True),
    addResolutions = cms.bool(False),
    beamLineSrc = cms.InputTag("offlineBeamSpot"),
    efficiencies = cms.PSet(

    ),
    electronSource = cms.InputTag("gedGsfElectrons"),
    embedBasicClusters = cms.bool(True),
    embedGenMatch = cms.bool(True),
    embedPreshowerClusters = cms.bool(True),
    embedRecHits = cms.bool(True),
    embedSeedCluster = cms.bool(True),
    embedSuperCluster = cms.bool(True),
    genParticleMatch = cms.InputTag("photonMatch"),
    isoDeposits = cms.PSet(
        pfChargedAll = cms.InputTag("phPFIsoDepositChargedAllPAT"),
        pfChargedHadrons = cms.InputTag("phPFIsoDepositChargedPAT"),
        pfNeutralHadrons = cms.InputTag("phPFIsoDepositNeutralPAT"),
        pfPUChargedHadrons = cms.InputTag("phPFIsoDepositPUPAT"),
        pfPhotons = cms.InputTag("phPFIsoDepositGammaPAT")
    ),
    isolationValues = cms.PSet(
        pfChargedAll = cms.InputTag("phPFIsoValueChargedAll04PFIdPAT"),
        pfChargedHadrons = cms.InputTag("phPFIsoValueCharged04PFIdPAT"),
        pfNeutralHadrons = cms.InputTag("phPFIsoValueNeutral04PFIdPAT"),
        pfPUChargedHadrons = cms.InputTag("phPFIsoValuePU04PFIdPAT"),
        pfPhotons = cms.InputTag("phPFIsoValueGamma04PFIdPAT")
    ),
    photonIDSources = cms.PSet(
        PhotonCutBasedIDLoose = cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDLoose"),
        PhotonCutBasedIDTight = cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDTight")
    ),
    photonSource = cms.InputTag("gedPhotons"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    resolutions = cms.PSet(

    ),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    ),
    userIsolation = cms.PSet(

    )
)


process.patTaus = cms.EDProducer("PATTauProducer",
    addEfficiencies = cms.bool(False),
    addGenJetMatch = cms.bool(True),
    addGenMatch = cms.bool(True),
    addResolutions = cms.bool(False),
    addTauID = cms.bool(True),
    addTauJetCorrFactors = cms.bool(False),
    efficiencies = cms.PSet(

    ),
    embedGenJetMatch = cms.bool(True),
    embedGenMatch = cms.bool(True),
    embedIsolationPFCands = cms.bool(False),
    embedIsolationPFChargedHadrCands = cms.bool(False),
    embedIsolationPFGammaCands = cms.bool(False),
    embedIsolationPFNeutralHadrCands = cms.bool(False),
    embedIsolationTracks = cms.bool(False),
    embedLeadPFCand = cms.bool(False),
    embedLeadPFChargedHadrCand = cms.bool(False),
    embedLeadPFNeutralCand = cms.bool(False),
    embedLeadTrack = cms.bool(False),
    embedSignalPFCands = cms.bool(False),
    embedSignalPFChargedHadrCands = cms.bool(False),
    embedSignalPFGammaCands = cms.bool(False),
    embedSignalPFNeutralHadrCands = cms.bool(False),
    embedSignalTracks = cms.bool(False),
    genJetMatch = cms.InputTag("tauGenJetMatch"),
    genParticleMatch = cms.InputTag("tauMatch"),
    isoDeposits = cms.PSet(
        pfAllParticles = cms.InputTag("tauIsoDepositPFCandidates"),
        pfChargedHadron = cms.InputTag("tauIsoDepositPFChargedHadrons"),
        pfGamma = cms.InputTag("tauIsoDepositPFGammas"),
        pfNeutralHadron = cms.InputTag("tauIsoDepositPFNeutralHadrons")
    ),
    resolutions = cms.PSet(

    ),
    tauIDSources = cms.PSet(
        againstElectronLoose = cms.InputTag("hpsPFTauDiscriminationByLooseElectronRejection"),
        againstElectronLooseMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5LooseElectronRejection"),
        againstElectronMVA5category = cms.InputTag("hpsPFTauDiscriminationByMVA5rawElectronRejection","category"),
        againstElectronMVA5raw = cms.InputTag("hpsPFTauDiscriminationByMVA5rawElectronRejection"),
        againstElectronMedium = cms.InputTag("hpsPFTauDiscriminationByMediumElectronRejection"),
        againstElectronMediumMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5MediumElectronRejection"),
        againstElectronTight = cms.InputTag("hpsPFTauDiscriminationByTightElectronRejection"),
        againstElectronTightMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5TightElectronRejection"),
        againstElectronVLooseMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5VLooseElectronRejection"),
        againstElectronVTightMVA5 = cms.InputTag("hpsPFTauDiscriminationByMVA5VTightElectronRejection"),
        againstMuonLoose = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection"),
        againstMuonLoose2 = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection2"),
        againstMuonLoose3 = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),
        againstMuonLooseMVA = cms.InputTag("hpsPFTauDiscriminationByMVALooseMuonRejection"),
        againstMuonMVAraw = cms.InputTag("hpsPFTauDiscriminationByMVArawMuonRejection"),
        againstMuonMedium = cms.InputTag("hpsPFTauDiscriminationByMediumMuonRejection"),
        againstMuonMedium2 = cms.InputTag("hpsPFTauDiscriminationByMediumMuonRejection2"),
        againstMuonMediumMVA = cms.InputTag("hpsPFTauDiscriminationByMVAMediumMuonRejection"),
        againstMuonTight = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection"),
        againstMuonTight2 = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection2"),
        againstMuonTight3 = cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection3"),
        againstMuonTightMVA = cms.InputTag("hpsPFTauDiscriminationByMVATightMuonRejection"),
        byCombinedIsolationDeltaBetaCorrRaw3Hits = cms.InputTag("hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits"),
        byIsolationMVA3newDMwLTraw = cms.InputTag("hpsPFTauDiscriminationByIsolationMVA3newDMwLTraw"),
        byIsolationMVA3newDMwoLTraw = cms.InputTag("hpsPFTauDiscriminationByIsolationMVA3newDMwoLTraw"),
        byIsolationMVA3oldDMwLTraw = cms.InputTag("hpsPFTauDiscriminationByIsolationMVA3oldDMwLTraw"),
        byIsolationMVA3oldDMwoLTraw = cms.InputTag("hpsPFTauDiscriminationByIsolationMVA3oldDMwoLTraw"),
        byLooseCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits"),
        byLooseIsolationMVA3newDMwLT = cms.InputTag("hpsPFTauDiscriminationByLooseIsolationMVA3newDMwLT"),
        byLooseIsolationMVA3newDMwoLT = cms.InputTag("hpsPFTauDiscriminationByLooseIsolationMVA3newDMwoLT"),
        byLooseIsolationMVA3oldDMwLT = cms.InputTag("hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwLT"),
        byLooseIsolationMVA3oldDMwoLT = cms.InputTag("hpsPFTauDiscriminationByLooseIsolationMVA3oldDMwoLT"),
        byMediumCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits"),
        byMediumIsolationMVA3newDMwLT = cms.InputTag("hpsPFTauDiscriminationByMediumIsolationMVA3newDMwLT"),
        byMediumIsolationMVA3newDMwoLT = cms.InputTag("hpsPFTauDiscriminationByMediumIsolationMVA3newDMwoLT"),
        byMediumIsolationMVA3oldDMwLT = cms.InputTag("hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwLT"),
        byMediumIsolationMVA3oldDMwoLT = cms.InputTag("hpsPFTauDiscriminationByMediumIsolationMVA3oldDMwoLT"),
        byTightCombinedIsolationDeltaBetaCorr3Hits = cms.InputTag("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
        byTightIsolationMVA3newDMwLT = cms.InputTag("hpsPFTauDiscriminationByTightIsolationMVA3newDMwLT"),
        byTightIsolationMVA3newDMwoLT = cms.InputTag("hpsPFTauDiscriminationByTightIsolationMVA3newDMwoLT"),
        byTightIsolationMVA3oldDMwLT = cms.InputTag("hpsPFTauDiscriminationByTightIsolationMVA3oldDMwLT"),
        byTightIsolationMVA3oldDMwoLT = cms.InputTag("hpsPFTauDiscriminationByTightIsolationMVA3oldDMwoLT"),
        byVLooseIsolationMVA3newDMwLT = cms.InputTag("hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwLT"),
        byVLooseIsolationMVA3newDMwoLT = cms.InputTag("hpsPFTauDiscriminationByVLooseIsolationMVA3newDMwoLT"),
        byVLooseIsolationMVA3oldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwLT"),
        byVLooseIsolationMVA3oldDMwoLT = cms.InputTag("hpsPFTauDiscriminationByVLooseIsolationMVA3oldDMwoLT"),
        byVTightIsolationMVA3newDMwLT = cms.InputTag("hpsPFTauDiscriminationByVTightIsolationMVA3newDMwLT"),
        byVTightIsolationMVA3newDMwoLT = cms.InputTag("hpsPFTauDiscriminationByVTightIsolationMVA3newDMwoLT"),
        byVTightIsolationMVA3oldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwLT"),
        byVTightIsolationMVA3oldDMwoLT = cms.InputTag("hpsPFTauDiscriminationByVTightIsolationMVA3oldDMwoLT"),
        byVVTightIsolationMVA3newDMwLT = cms.InputTag("hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwLT"),
        byVVTightIsolationMVA3newDMwoLT = cms.InputTag("hpsPFTauDiscriminationByVVTightIsolationMVA3newDMwoLT"),
        byVVTightIsolationMVA3oldDMwLT = cms.InputTag("hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwLT"),
        byVVTightIsolationMVA3oldDMwoLT = cms.InputTag("hpsPFTauDiscriminationByVVTightIsolationMVA3oldDMwoLT"),
        chargedIsoPtSum = cms.InputTag("hpsPFTauMVA3IsolationChargedIsoPtSum"),
        decayModeFinding = cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
        decayModeFindingNewDMs = cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),
        neutralIsoPtSum = cms.InputTag("hpsPFTauMVA3IsolationNeutralIsoPtSum"),
        puCorrPtSum = cms.InputTag("hpsPFTauMVA3IsolationPUcorrPtSum")
    ),
    tauJetCorrFactorsSource = cms.VInputTag(cms.InputTag("patTauJetCorrFactors")),
    tauSource = cms.InputTag("hpsPFTauProducer"),
    tauTransverseImpactParameterSource = cms.InputTag("hpsPFTauTransverseImpactParameters"),
    userData = cms.PSet(
        userCands = cms.PSet(
            src = cms.VInputTag("")
        ),
        userClasses = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFloats = cms.PSet(
            src = cms.VInputTag("")
        ),
        userFunctionLabels = cms.vstring(),
        userFunctions = cms.vstring(),
        userInts = cms.PSet(
            src = cms.VInputTag("")
        )
    ),
    userIsolation = cms.PSet(
        pfAllParticles = cms.PSet(
            deltaR = cms.double(0.5),
            src = cms.InputTag("tauIsoDepositPFCandidates"),
            threshold = cms.double(0.0)
        ),
        pfChargedHadron = cms.PSet(
            deltaR = cms.double(0.5),
            src = cms.InputTag("tauIsoDepositPFChargedHadrons"),
            threshold = cms.double(0.0)
        ),
        pfGamma = cms.PSet(
            deltaR = cms.double(0.5),
            src = cms.InputTag("tauIsoDepositPFGammas"),
            threshold = cms.double(0.0)
        ),
        pfNeutralHadron = cms.PSet(
            deltaR = cms.double(0.5),
            src = cms.InputTag("tauIsoDepositPFNeutralHadrons"),
            threshold = cms.double(0.0)
        )
    )
)


process.pfBoostedDoubleSecondaryVertexAK8BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateBoostedDoubleSecondaryVertexAK8Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfosAK8"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfosAK8"), cms.InputTag("softPFMuonsTagInfosAK8"), cms.InputTag("softPFElectronsTagInfosAK8"))
)


process.pfBoostedDoubleSecondaryVertexCA15BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateBoostedDoubleSecondaryVertexCA15Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfosCA15"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfosCA15"), cms.InputTag("softPFMuonsTagInfosCA15"), cms.InputTag("softPFElectronsTagInfosCA15"))
)


process.pfCandMETcorr = cms.EDProducer("PFCandMETcorrInputProducer",
    src = cms.InputTag("pfCandsNotInJetsForMetCorr")
)


process.pfCandsNotInJetsForMetCorr = cms.EDProducer("PFCandidateFromFwdPtrProducer",
    src = cms.InputTag("pfCandsNotInJetsPtrForMetCorr")
)


process.pfCandsNotInJetsPtrForMetCorr = cms.EDProducer("TPPFJetsOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlowPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('noJet'),
    topCollection = cms.InputTag("pfJetsPtrForMetCorr"),
    verbose = cms.untracked.bool(False)
)


process.pfCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)


process.pfCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)


process.pfCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexTagInfos"))
)


process.pfCombinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexSoftLeptonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfCombinedSecondaryVertexSoftLeptonCtagLJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexSoftLeptonCtagLComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderCtagLTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexTagInfos"))
)


process.pfImpactParameterTagInfos = cms.EDProducer("CandIPProducer",
    candidates = cms.InputTag("particleFlow"),
    computeGhostTrack = cms.bool(True),
    computeProbabilities = cms.bool(True),
    ghostTrackPriorDeltaR = cms.double(0.03),
    jetDirectionUsingGhostTrack = cms.bool(False),
    jetDirectionUsingTracks = cms.bool(False),
    jets = cms.InputTag("ak4PFJetsCHS"),
    maxDeltaR = cms.double(0.4),
    maximumChiSquared = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumNumberOfPixelHits = cms.int32(2),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    useTrackQuality = cms.bool(False)
)


process.pfImpactParameterTagInfosAK8 = cms.EDProducer("CandIPProducer",
    candidates = cms.InputTag("particleFlow"),
    computeGhostTrack = cms.bool(False),
    computeProbabilities = cms.bool(False),
    ghostTrackPriorDeltaR = cms.double(0.03),
    jetDirectionUsingGhostTrack = cms.bool(False),
    jetDirectionUsingTracks = cms.bool(False),
    jets = cms.InputTag("ak8PFJetsCHS"),
    maxDeltaR = cms.double(0.8),
    maximumChiSquared = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumNumberOfPixelHits = cms.int32(2),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    useTrackQuality = cms.bool(False)
)


process.pfImpactParameterTagInfosCA15 = cms.EDProducer("CandIPProducer",
    candidates = cms.InputTag("particleFlow"),
    computeGhostTrack = cms.bool(False),
    computeProbabilities = cms.bool(False),
    ghostTrackPriorDeltaR = cms.double(0.03),
    jetDirectionUsingGhostTrack = cms.bool(False),
    jetDirectionUsingTracks = cms.bool(False),
    jets = cms.InputTag("ca15PFJetsCHS"),
    maxDeltaR = cms.double(1.5),
    maximumChiSquared = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumNumberOfPixelHits = cms.int32(2),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    useTrackQuality = cms.bool(False)
)


process.pfInclusiveSecondaryVertexFinderCtagLNegativeTagInfos = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveCandidateSecondaryVerticesCtagL"),
    extSVDeltaRToJet = cms.double(-0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(-2.0),
        distSig2dMin = cms.double(-99999.9),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(-0.01),
        distVal2dMin = cms.double(-2.5),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(-0.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfInclusiveSecondaryVertexFinderCtagLTagInfos = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveCandidateSecondaryVerticesCtagL"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(1.5),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfInclusiveSecondaryVertexFinderNegativeTagInfos = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    extSVDeltaRToJet = cms.double(-0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(-2.0),
        distSig2dMin = cms.double(-99999.9),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(-0.01),
        distVal2dMin = cms.double(-2.5),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(-0.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfInclusiveSecondaryVertexFinderTagInfos = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(2.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfInclusiveSecondaryVertexFinderTagInfosAK8 = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    extSVDeltaRToJet = cms.double(0.8),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfosAK8"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.8),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(2.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.8),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfInclusiveSecondaryVertexFinderTagInfosCA15 = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    extSVDeltaRToJet = cms.double(1.5),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfosCA15"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(1.5),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(True),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(2.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.79),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(1.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfJetsPtrForMetCorr = cms.EDProducer("PFJetFwdPtrProducer",
    src = cms.InputTag("ak4PFJets")
)


process.pfMetT0pc = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0PfCand"))
)


process.pfMetT0pcT1 = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0PfCand"), cms.InputTag("corrPfMetType1","type1"))
)


process.pfMetT0pcT1T2Txy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0PfCand"), cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetType2"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT0pcT1Txy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0PfCand"), cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT0pcTxy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0PfCand"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT0rt = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrack"))
)


process.pfMetT0rtT1 = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrack"), cms.InputTag("corrPfMetType1","type1"))
)


process.pfMetT0rtT1T2 = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrackForType2"), cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetType2"))
)


process.pfMetT0rtT1T2Txy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrackForType2"), cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetType2"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT0rtT1Txy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrack"), cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT0rtT2 = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrackForType2"), cms.InputTag("corrPfMetType2"))
)


process.pfMetT0rtTxy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType0RecoTrack"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT1 = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType1","type1"))
)


process.pfMetT1T2 = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetType2"))
)


process.pfMetT1T2Txy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetType2"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetT1Txy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetType1","type1"), cms.InputTag("corrPfMetXYMult"))
)


process.pfMetTxy = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("pfMet"),
    srcCorrections = cms.VInputTag(cms.InputTag("corrPfMetXYMult"))
)


process.pfNegativeCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"))
)


process.pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"))
)


process.pfNegativeCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfNegativeCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)


process.pfNegativeCombinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexSoftLeptonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderNegativeTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfNegativeCombinedSecondaryVertexSoftLeptonCtagLJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexSoftLeptonCtagLComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderCtagLNegativeTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfNegativeCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)


process.pfNegativeOnlyJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeOnlyJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfNegativeOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeOnlyJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfNegativeSimpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateSimpleSecondaryVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)


process.pfNegativeSimpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateSimpleSecondaryVertex3TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexNegativeTagInfos"))
)


process.pfNegativeTrackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeTrackCounting3D2ndComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfNegativeTrackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeTrackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfNoJet = cms.EDProducer("TPPFJetsOnPFCandidates",
    bottomCollection = cms.InputTag("pfNoElectronJME"),
    enable = cms.bool(True),
    name = cms.untracked.string('noJet'),
    topCollection = cms.InputTag("pfJetsPtrs"),
    verbose = cms.untracked.bool(False)
)


process.pfNoPileUp = cms.EDProducer("TPPFCandidatesOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlowTmpPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('pileUpOnPFCandidates'),
    topCollection = cms.InputTag("pfPileUp"),
    verbose = cms.untracked.bool(False)
)


process.pfNoPileUpIsoPFBRECO = cms.EDProducer("TPPFCandidatesOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlowPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('pileUpOnPFCandidates'),
    topCollection = cms.InputTag("pfPileUpIsoPFBRECO"),
    verbose = cms.untracked.bool(False)
)


process.pfNoPileUpJME = cms.EDProducer("TPPFCandidatesOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlowPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('pileUpOnPFCandidates'),
    topCollection = cms.InputTag("pfPileUpJME"),
    verbose = cms.untracked.bool(False)
)


process.pfNoPileUpPFBRECO = cms.EDProducer("TPPFCandidatesOnPFCandidates",
    bottomCollection = cms.InputTag("particleFlowPtrs"),
    enable = cms.bool(True),
    name = cms.untracked.string('pileUpOnPFCandidates'),
    topCollection = cms.InputTag("pfPileUpPFBRECO"),
    verbose = cms.untracked.bool(False)
)


process.pfPileUp = cms.EDProducer("PFPileUp",
    Enable = cms.bool(True),
    PFCandidates = cms.InputTag("particleFlowTmpPtrs"),
    Vertices = cms.InputTag("offlinePrimaryVertices"),
    checkClosestZVertex = cms.bool(True),
    verbose = cms.untracked.bool(False)
)


process.pfPileUpIsoPFBRECO = cms.EDProducer("PFPileUp",
    Enable = cms.bool(True),
    PFCandidates = cms.InputTag("particleFlowPtrs"),
    Vertices = cms.InputTag("offlinePrimaryVertices"),
    checkClosestZVertex = cms.bool(True),
    verbose = cms.untracked.bool(False)
)


process.pfPileUpJME = cms.EDProducer("PFPileUp",
    Enable = cms.bool(True),
    PFCandidates = cms.InputTag("particleFlowPtrs"),
    Vertices = cms.InputTag("goodOfflinePrimaryVertices"),
    checkClosestZVertex = cms.bool(False),
    verbose = cms.untracked.bool(False)
)


process.pfPileUpPFBRECO = cms.EDProducer("PFPileUp",
    Enable = cms.bool(True),
    PFCandidates = cms.InputTag("particleFlowPtrs"),
    Vertices = cms.InputTag("offlinePrimaryVertices"),
    checkClosestZVertex = cms.bool(True),
    verbose = cms.untracked.bool(False)
)


process.pfPositiveCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)


process.pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"))
)


process.pfPositiveCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfPositiveCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexTagInfos"))
)


process.pfPositiveCombinedSecondaryVertexSoftLeptonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexSoftLeptonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfPositiveCombinedSecondaryVertexSoftLeptonCtagLJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexSoftLeptonCtagLComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfInclusiveSecondaryVertexFinderCtagLTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.pfPositiveCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"), cms.InputTag("pfSecondaryVertexTagInfos"))
)


process.pfPositiveOnlyJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveOnlyJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfPositiveOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveOnlyJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfSecondaryVertexNegativeTagInfos = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("secondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(False),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(-3.0),
        distSig2dMin = cms.double(-99999.9),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(-0.01),
        distVal2dMin = cms.double(-2.5),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(-0.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfSecondaryVertexTagInfos = cms.EDProducer("CandSecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("secondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("pfImpactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(False),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(3.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.pfSimpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateSimpleSecondaryVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos"))
)


process.pfSimpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateSimpleSecondaryVertex3TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos"))
)


process.pfTrackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateTrackCounting3D2ndComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.pfTrackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateTrackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


process.phPFIsoDepositChargedAllPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedPhotons"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositChargedAllPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedPhotonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositChargedPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedPhotons"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositChargedPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllChargedHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedPhotonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositGammaPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        MissHitSCMatch_Veto = cms.bool(False),
        SCMatch_Veto = cms.bool(True),
        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedPhotons"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositGammaPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        MissHitSCMatch_Veto = cms.bool(False),
        SCMatch_Veto = cms.bool(True),
        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedPhotonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositNeutralPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllNeutralHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedPhotons"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositNeutralPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfAllNeutralHadronsPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedPhotonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositPUPAT = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfPileUpAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("gedPhotons"),
    trackType = cms.string('candidate')
)


process.phPFIsoDepositPUPFBRECO = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('CandViewExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(99999.99),
        Diff_z = cms.double(99999.99),
        inputCandView = cms.InputTag("pfPileUpAllChargedParticlesPFBRECO")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("pfSelectedPhotonsPFBRECO"),
    trackType = cms.string('candidate')
)


process.phPFIsoValueCharged03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositCharged"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueCharged03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueCharged03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueCharged04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositCharged"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueCharged04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueCharged04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueChargedAll03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedAll"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueChargedAll03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueChargedAll03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueChargedAll04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedAll"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueChargedAll04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedAllPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueChargedAll04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositChargedAllPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueGamma03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositGamma"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueGamma03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositGammaPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueGamma03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueGamma04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositGamma"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueGamma04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositGammaPAT"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueGamma04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositGammaPFBRECO"),
        vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueNeutral03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositNeutral"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueNeutral03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositNeutralPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueNeutral03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueNeutral04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositNeutral"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueNeutral04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositNeutralPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValueNeutral04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositNeutralPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValuePU03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositPU"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValuePU03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositPUPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValuePU03PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.3),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValuePU04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositPU"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValuePU04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositPUPAT"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.phPFIsoValuePU04PFIdPFBRECO = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        PivotCoordinatesForEBEE = cms.bool(True),
        deltaR = cms.double(0.4),
        mode = cms.string('sum'),
        skipDefaultVeto = cms.bool(True),
        src = cms.InputTag("phPFIsoDepositPUPFBRECO"),
        vetos = cms.vstring(),
        weight = cms.string('1')
    ))
)


process.photonMatch = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(True),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    mcPdgId = cms.vint32(22),
    mcStatus = cms.vint32(1),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("gedPhotons")
)


process.pileupJetIdCalculator = cms.EDProducer("PileupJetIdProducer",
    algos = cms.VPSet(cms.PSet(
        JetIdParams = cms.PSet(
            Pt010_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt010_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt010_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt010_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
            Pt010_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt010_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
            Pt1020_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt1020_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt1020_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt1020_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
            Pt1020_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt1020_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
            Pt2030_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt2030_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt2030_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt2030_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
            Pt2030_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt2030_RMSTight = cms.vdouble(0.05, 0.07, 0.03, 0.045),
            Pt3050_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt3050_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt3050_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt3050_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
            Pt3050_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt3050_RMSTight = cms.vdouble(0.05, 0.06, 0.03, 0.04)
        ),
        cutBased = cms.bool(True),
        impactParTkThreshold = cms.double(1.0),
        label = cms.string('cutbased')
    )),
    applyJec = cms.bool(False),
    inputIsCorrected = cms.bool(True),
    jec = cms.string('AK4PFchs'),
    jetids = cms.InputTag(""),
    jets = cms.InputTag("ak4PFJetsCHS"),
    produceJetIds = cms.bool(True),
    residualsFromTxt = cms.bool(False),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    runMvas = cms.bool(False),
    vertexes = cms.InputTag("offlinePrimaryVertices")
)


process.pileupJetIdEvaluator = cms.EDProducer("PileupJetIdProducer",
    algos = cms.VPSet(cms.PSet(
        JetIdParams = cms.PSet(
            Pt010_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt010_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt010_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt010_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
            Pt010_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt010_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
            Pt1020_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt1020_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt1020_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt1020_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
            Pt1020_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt1020_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
            Pt2030_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt2030_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt2030_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt2030_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
            Pt2030_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt2030_RMSTight = cms.vdouble(0.05, 0.07, 0.03, 0.045),
            Pt3050_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt3050_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
            Pt3050_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
            Pt3050_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
            Pt3050_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
            Pt3050_RMSTight = cms.vdouble(0.05, 0.06, 0.03, 0.04)
        ),
        cutBased = cms.bool(True),
        impactParTkThreshold = cms.double(1.0),
        label = cms.string('cutbased')
    ), 
        cms.PSet(
            JetIdParams = cms.PSet(
                Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
                Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
                Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
                Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
                Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
                Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
                Pt2030_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
                Pt2030_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
                Pt2030_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17),
                Pt3050_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
                Pt3050_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
                Pt3050_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17)
            ),
            cutBased = cms.bool(False),
            impactParTkThreshold = cms.double(1.0),
            label = cms.string('full'),
            tmvaMethod = cms.string('JetIDMVAHighPt'),
            tmvaSpectators = cms.vstring('p4.fCoordinates.fPt', 
                'p4.fCoordinates.fEta', 
                'nTrueInt', 
                'dRMatch'),
            tmvaVariables = cms.vstring('DRweighted', 
                'rho', 
                'nTot', 
                'nCh', 
                'axisMajor', 
                'axisMinor', 
                'fRing0', 
                'fRing1', 
                'fRing2', 
                'fRing3', 
                'ptD', 
                'beta', 
                'betaStar', 
                'min(pull,0.1)', 
                'jetR', 
                'jetRchg'),
            tmvaWeights_jteta_0_2 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_0_2.xml'),
            tmvaWeights_jteta_2_2p5 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_2_2p5.xml'),
            tmvaWeights_jteta_2p5_3 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_2p5_3.xml'),
            tmvaWeights_jteta_3_5 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_3_5.xml'),
            version = cms.int32(-1)
        )),
    applyJec = cms.bool(False),
    inputIsCorrected = cms.bool(True),
    jec = cms.string('AK4PFchs'),
    jetids = cms.InputTag("pileupJetIdCalculator"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    produceJetIds = cms.bool(False),
    residualsFromTxt = cms.bool(False),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    runMvas = cms.bool(True),
    vertexes = cms.InputTag("offlinePrimaryVertices")
)


process.positiveCombinedInclusiveSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)


process.positiveCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)


process.positiveCombinedMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveCombinedMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"), cms.InputTag("softPFMuonsTagInfos"), cms.InputTag("softPFElectronsTagInfos"))
)


process.positiveCombinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveCombinedSecondaryVertexComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexTagInfos"))
)


process.positiveCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveCombinedSecondaryVertexV2Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"), cms.InputTag("secondaryVertexTagInfos"))
)


process.positiveOnlyJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveOnlyJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.positiveOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveOnlyJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.positiveSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.positiveSoftPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.positiveSoftPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.positiveSoftPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.positiveSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.positiveSoftPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.positiveSoftPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.positiveSoftPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.puppi = cms.EDProducer("PuppiProducer",
    DeltaZCut = cms.double(0.3),
    MinPuppiWeight = cms.double(0.01),
    UseDeltaZCut = cms.bool(False),
    algos = cms.VPSet(cms.PSet(
        EtaMaxExtrap = cms.double(2.0),
        MedEtaSF = cms.double(1.0),
        MinNeutralPt = cms.double(0.1),
        MinNeutralPtSlope = cms.double(0.005),
        RMSEtaSF = cms.double(1.0),
        etaMax = cms.double(2.0),
        etaMin = cms.double(0.0),
        ptMin = cms.double(0.0),
        puppiAlgos = cms.VPSet(cms.PSet(
            algoId = cms.int32(5),
            applyLowPUCorr = cms.bool(True),
            combOpt = cms.int32(0),
            cone = cms.double(0.3),
            rmsPtMin = cms.double(0.1),
            rmsScaleFactor = cms.double(1.0),
            useCharged = cms.bool(True)
        ))
    ), 
        cms.PSet(
            EtaMaxExtrap = cms.double(2.0),
            MedEtaSF = cms.double(1.0),
            MinNeutralPt = cms.double(0.1),
            MinNeutralPtSlope = cms.double(0.005),
            RMSEtaSF = cms.double(0.95),
            etaMax = cms.double(2.5),
            etaMin = cms.double(2.0),
            ptMin = cms.double(0.0),
            puppiAlgos = cms.VPSet(cms.PSet(
                algoId = cms.int32(5),
                applyLowPUCorr = cms.bool(True),
                combOpt = cms.int32(0),
                cone = cms.double(0.3),
                rmsPtMin = cms.double(0.1),
                rmsScaleFactor = cms.double(1.0),
                useCharged = cms.bool(True)
            ))
        ), 
        cms.PSet(
            EtaMaxExtrap = cms.double(2.0),
            MedEtaSF = cms.double(0.845),
            MinNeutralPt = cms.double(1.0),
            MinNeutralPtSlope = cms.double(0.005),
            RMSEtaSF = cms.double(1.484),
            etaMax = cms.double(3.0),
            etaMin = cms.double(2.5),
            ptMin = cms.double(0.0),
            puppiAlgos = cms.VPSet(cms.PSet(
                algoId = cms.int32(5),
                applyLowPUCorr = cms.bool(True),
                combOpt = cms.int32(0),
                cone = cms.double(0.3),
                rmsPtMin = cms.double(0.1),
                rmsScaleFactor = cms.double(1.0),
                useCharged = cms.bool(False)
            ))
        ), 
        cms.PSet(
            EtaMaxExtrap = cms.double(2.0),
            MedEtaSF = cms.double(0.4397),
            MinNeutralPt = cms.double(1.0),
            MinNeutralPtSlope = cms.double(0.005),
            RMSEtaSF = cms.double(1.25),
            etaMax = cms.double(10.0),
            etaMin = cms.double(3.0),
            ptMin = cms.double(0.0),
            puppiAlgos = cms.VPSet(cms.PSet(
                algoId = cms.int32(5),
                applyLowPUCorr = cms.bool(True),
                combOpt = cms.int32(0),
                cone = cms.double(0.3),
                rmsPtMin = cms.double(0.1),
                rmsScaleFactor = cms.double(1.0),
                useCharged = cms.bool(False)
            ))
        )),
    applyCHS = cms.bool(True),
    candName = cms.InputTag("particleFlow"),
    puppiDiagnostics = cms.bool(False),
    useExp = cms.bool(False),
    vertexName = cms.InputTag("offlinePrimaryVertices")
)


process.secondaryVertexNegativeTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("secondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(False),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(-3.0),
        distSig2dMin = cms.double(-99999.9),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(-0.01),
        distVal2dMin = cms.double(-2.5),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(-0.5),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.secondaryVertexTagInfos = cms.EDProducer("SecondaryVertexProducer",
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string('BeamSpot'),
    extSVCollection = cms.InputTag("secondaryVertices"),
    extSVDeltaRToJet = cms.double(0.3),
    minimumTrackWeight = cms.double(0.5),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip3dSig'),
    useExternalSV = cms.bool(False),
    usePVError = cms.bool(True),
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(3.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    ),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)


process.simpleInclusiveSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredTagInfos"))
)


process.simpleInclusiveSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("inclusiveSecondaryVertexFinderFilteredTagInfos"))
)


process.simpleSecondaryVertexHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex2TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)


process.simpleSecondaryVertexHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('simpleSecondaryVertex3TrkComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("secondaryVertexTagInfos"))
)


process.sisCone5GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.sisCone5GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.sisCone5GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.sisCone5PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.5),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.sisCone7PFJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.7),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)


process.sisCone8GenJets = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.sisCone8GenJetsNoMuNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoMuNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.sisCone8GenJetsNoNu = cms.EDProducer("FastjetJetProducer",
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetAlgorithm = cms.string('SISCone'),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    rParam = cms.double(0.8),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJetsNoNu"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)


process.softKiller = cms.EDProducer("SoftKillerProducer",
    PFCandidates = cms.InputTag("particleFlow"),
    Rho_EtaMax = cms.double(5.0),
    rParam = cms.double(0.4)
)


process.softMuonTagInfos = cms.EDProducer("SoftLepton",
    jets = cms.InputTag("ak4PFJetsCHS"),
    leptonCands = cms.InputTag(""),
    leptonChi2Cut = cms.double(9999.0),
    leptonDeltaRCut = cms.double(0.4),
    leptonId = cms.InputTag(""),
    leptons = cms.InputTag("muons"),
    muonSelection = cms.uint32(1),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    refineJetAxis = cms.uint32(0)
)


process.softPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.softPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.softPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.softPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)


process.softPFElectronsTagInfos = cms.EDProducer("SoftPFElectronTagInfoProducer",
    DeltaRElectronJet = cms.double(0.4),
    MaxSip3D = cms.double(200),
    electrons = cms.InputTag("gedGsfElectrons"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices")
)


process.softPFElectronsTagInfosAK8 = cms.EDProducer("SoftPFElectronTagInfoProducer",
    DeltaRElectronJet = cms.double(0.8),
    MaxSip3D = cms.double(200),
    electrons = cms.InputTag("gedGsfElectrons"),
    jets = cms.InputTag("ak8PFJetsCHS"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices")
)


process.softPFElectronsTagInfosCA15 = cms.EDProducer("SoftPFElectronTagInfoProducer",
    DeltaRElectronJet = cms.double(1.5),
    MaxSip3D = cms.double(200),
    electrons = cms.InputTag("gedGsfElectrons"),
    jets = cms.InputTag("ca15PFJetsCHS"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices")
)


process.softPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.softPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.softPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.softPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)


process.softPFMuonsTagInfos = cms.EDProducer("SoftPFMuonTagInfoProducer",
    filterIp = cms.double(4.0),
    filterPromptMuons = cms.bool(False),
    filterRatio1 = cms.double(0.4),
    filterRatio2 = cms.double(0.7),
    jets = cms.InputTag("ak4PFJetsCHS"),
    muonPt = cms.double(2.0),
    muonSIP = cms.double(200.0),
    muons = cms.InputTag("muons"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices")
)


process.softPFMuonsTagInfosAK8 = cms.EDProducer("SoftPFMuonTagInfoProducer",
    filterIp = cms.double(4.0),
    filterPromptMuons = cms.bool(False),
    filterRatio1 = cms.double(0.4),
    filterRatio2 = cms.double(0.7),
    jets = cms.InputTag("ak8PFJetsCHS"),
    muonPt = cms.double(2.0),
    muonSIP = cms.double(200.0),
    muons = cms.InputTag("muons"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices")
)


process.softPFMuonsTagInfosCA15 = cms.EDProducer("SoftPFMuonTagInfoProducer",
    filterIp = cms.double(4.0),
    filterPromptMuons = cms.bool(False),
    filterRatio1 = cms.double(0.4),
    filterRatio2 = cms.double(0.7),
    jets = cms.InputTag("ca15PFJetsCHS"),
    muonPt = cms.double(2.0),
    muonSIP = cms.double(200.0),
    muons = cms.InputTag("muons"),
    primaryVertex = cms.InputTag("offlinePrimaryVertices")
)


process.tauGenJetMatch = cms.EDProducer("GenJetMatcher",
    checkCharge = cms.bool(False),
    matched = cms.InputTag("tauGenJetsSelectorAllHadrons"),
    maxDPtRel = cms.double(3.0),
    maxDeltaR = cms.double(0.1),
    mcPdgId = cms.vint32(),
    mcStatus = cms.vint32(),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("hpsPFTauProducer")
)


process.tauGenJets = cms.EDProducer("TauGenJetProducer",
    GenParticles = cms.InputTag("genParticles"),
    includeNeutrinos = cms.bool(False),
    verbose = cms.untracked.bool(False)
)


process.tauIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFTauExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0.0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(10000.0),
        Diff_z = cms.double(10000.0),
        candidateSource = cms.InputTag("particleFlow"),
        dRmatchPFTau = cms.double(0.1),
        dRvetoPFTauSignalConeConstituents = cms.double(0.01),
        tauSource = cms.InputTag("hpsPFTauProducer")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("hpsPFTauProducer"),
    trackType = cms.string('candidate')
)


process.tauIsoDepositPFChargedHadrons = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFTauExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0.0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(0.1),
        Diff_z = cms.double(0.2),
        candidateSource = cms.InputTag("pfAllChargedHadronsPFBRECO"),
        dRmatchPFTau = cms.double(0.1),
        dRvetoPFTauSignalConeConstituents = cms.double(0.01),
        tauSource = cms.InputTag("hpsPFTauProducer")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("hpsPFTauProducer"),
    trackType = cms.string('candidate')
)


process.tauIsoDepositPFGammas = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFTauExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0.0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(10000.0),
        Diff_z = cms.double(10000.0),
        candidateSource = cms.InputTag("pfAllPhotonsPFBRECO"),
        dRmatchPFTau = cms.double(0.1),
        dRvetoPFTauSignalConeConstituents = cms.double(0.01),
        tauSource = cms.InputTag("hpsPFTauProducer")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("hpsPFTauProducer"),
    trackType = cms.string('candidate')
)


process.tauIsoDepositPFNeutralHadrons = cms.EDProducer("CandIsoDepositProducer",
    ExtractorPSet = cms.PSet(
        ComponentName = cms.string('PFTauExtractor'),
        DR_Max = cms.double(1.0),
        DR_Veto = cms.double(0.0),
        DepositLabel = cms.untracked.string(''),
        Diff_r = cms.double(10000.0),
        Diff_z = cms.double(10000.0),
        candidateSource = cms.InputTag("pfAllNeutralHadronsPFBRECO"),
        dRmatchPFTau = cms.double(0.1),
        dRvetoPFTauSignalConeConstituents = cms.double(0.01),
        tauSource = cms.InputTag("hpsPFTauProducer")
    ),
    MultipleDepositsFlag = cms.bool(False),
    src = cms.InputTag("hpsPFTauProducer"),
    trackType = cms.string('candidate')
)


process.tauMatch = cms.EDProducer("MCMatcher",
    checkCharge = cms.bool(True),
    matched = cms.InputTag("genParticles"),
    maxDPtRel = cms.double(999.9),
    maxDeltaR = cms.double(999.9),
    mcPdgId = cms.vint32(15),
    mcStatus = cms.vint32(2),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
    src = cms.InputTag("hpsPFTauProducer")
)


process.trackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('trackCounting3D2ndComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.trackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('trackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


process.ak8PFJetsCHSConstituents = cms.EDFilter("PFJetConstituentSelector",
    cut = cms.string('pt > 100.0 && abs(rapidity()) < 2.4'),
    src = cms.InputTag("ak8PFJetsCHS")
)


process.ak8PFJetsCSConstituents = cms.EDFilter("PFJetConstituentSelector",
    cut = cms.string('pt > 100.0'),
    src = cms.InputTag("ak8PFJetsCS")
)


process.bVertexFilter = cms.EDFilter("BVertexFilter",
    minVertices = cms.int32(0),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    secondaryVertices = cms.InputTag("secondaryVertices"),
    useVertexKinematicAsJetAxis = cms.bool(True),
    vertexFilter = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(3.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.1),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    )
)


process.goodOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
    cut = cms.string('!isFake && ndof >= 4.0 && abs(z) <= 24.0 && abs(position.Rho) <= 2.0'),
    filter = cms.bool(False),
    src = cms.InputTag("offlinePrimaryVertices")
)


process.inclusiveSecondaryVerticesFiltered = cms.EDFilter("BVertexFilter",
    minVertices = cms.int32(0),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    secondaryVertices = cms.InputTag("inclusiveSecondaryVertices"),
    useVertexKinematicAsJetAxis = cms.bool(True),
    vertexFilter = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(3.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.1),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    )
)


process.pfAllChargedHadronsPFBRECO = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    makeClones = cms.bool(True),
    pdgId = cms.vint32(211, -211, 321, -321, 999211, 
        2212, -2212),
    src = cms.InputTag("pfNoPileUpIsoPFBRECO")
)


process.pfAllChargedParticlesPFBRECO = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    makeClones = cms.bool(True),
    pdgId = cms.vint32(211, -211, 321, -321, 999211, 
        2212, -2212, 11, -11, 13, 
        -13),
    src = cms.InputTag("pfNoPileUpIsoPFBRECO")
)


process.pfAllNeutralHadronsAndPhotonsPFBRECO = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    makeClones = cms.bool(True),
    pdgId = cms.vint32(22, 111, 130, 310, 2112),
    src = cms.InputTag("pfNoPileUpIsoPFBRECO")
)


process.pfAllNeutralHadronsPFBRECO = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    makeClones = cms.bool(True),
    pdgId = cms.vint32(111, 130, 310, 2112),
    src = cms.InputTag("pfNoPileUpIsoPFBRECO")
)


process.pfAllPhotonsPFBRECO = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    makeClones = cms.bool(True),
    pdgId = cms.vint32(22),
    src = cms.InputTag("pfNoPileUpIsoPFBRECO")
)


process.pfPileUpAllChargedParticlesPFBRECO = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    makeClones = cms.bool(True),
    pdgId = cms.vint32(211, -211, 321, -321, 999211, 
        2212, -2212, 11, -11, 13, 
        -13),
    src = cms.InputTag("pfPileUpIsoPFBRECO")
)


process.selectedPatElectrons = cms.EDFilter("PATElectronSelector",
    cut = cms.string(''),
    src = cms.InputTag("patElectrons")
)


process.selectedPatJets = cms.EDFilter("PATJetSelector",
    cut = cms.string(''),
    src = cms.InputTag("patJets")
)


process.selectedPatJetsAK4PFCHS = cms.EDFilter("PATJetSelector",
    cut = cms.string(''),
    src = cms.InputTag("patJetsAK4PFCHS")
)


process.selectedPatJetsAK8PFCHS = cms.EDFilter("PATJetSelector",
    cut = cms.string(''),
    src = cms.InputTag("patJetsAK8PFCHS")
)


process.selectedPatJetsCA8PFCHS = cms.EDFilter("PATJetSelector",
    cut = cms.string(''),
    src = cms.InputTag("patJetsCA8PFCHS")
)


process.selectedPatMuons = cms.EDFilter("PATMuonSelector",
    cut = cms.string(''),
    src = cms.InputTag("patMuons")
)


process.selectedPatPhotons = cms.EDFilter("PATPhotonSelector",
    cut = cms.string(''),
    src = cms.InputTag("patPhotons")
)


process.selectedPatTaus = cms.EDFilter("PATTauSelector",
    cut = cms.string(''),
    src = cms.InputTag("patTaus")
)


process.tauGenJetsSelectorAllHadrons = cms.EDFilter("TauGenJetDecayModeSelector",
    filter = cms.bool(False),
    select = cms.vstring('oneProng0Pi0', 
        'oneProng1Pi0', 
        'oneProng2Pi0', 
        'oneProngOther', 
        'threeProng0Pi0', 
        'threeProng1Pi0', 
        'threeProngOther', 
        'rare'),
    src = cms.InputTag("tauGenJets")
)


process.patCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    candidates = cms.VInputTag(cms.InputTag("patElectrons"), cms.InputTag("patMuons"), cms.InputTag("patTaus"), cms.InputTag("patPhotons"), cms.InputTag("patJets"), 
        cms.InputTag("patMETs")),
    logName = cms.untracked.string('patCandidates|PATSummaryTables')
)


process.selectedPatCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    candidates = cms.VInputTag(cms.InputTag("selectedPatElectrons"), cms.InputTag("selectedPatMuons"), cms.InputTag("selectedPatTaus"), cms.InputTag("selectedPatPhotons"), cms.InputTag("selectedPatJets")),
    logName = cms.untracked.string('selectedPatCanddiates|PATSummaryTables')
)


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testJetTools.root'),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_selectedPatPhotons*_*_*', 
        'keep *_selectedPatElectrons*_*_*', 
        'keep *_selectedPatMuons*_*_*', 
        'keep *_selectedPatTaus*_*_*', 
        'keep *_selectedPatJets*_*_*', 
        'drop *_*PF_caloTowers_*', 
        'drop *_*JPT_pfCandidates_*', 
        'drop *_*Calo_pfCandidates_*', 
        'keep *_patMETs*_*_*', 
        'keep *_selectedPatPFParticles*_*_*', 
        'keep *_selectedPatTrackCands*_*_*', 
        'drop *_selectedAK4PFCHS_tagInfos_*', 
        'drop *_selectedCA8PFCHS_tagInfos_*', 
        'drop *_selectedAK8PFCHS_tagInfos_*', 
        'keep *_ak4PFJetsCHS_*_*', 
        'keep *_patJetsAK4PFCHS_*_*', 
        'keep *_ca8PFJetsCHS_*_*', 
        'keep *_patJetsCA8PFCHS_*_*', 
        'keep *_ak8PFJetsCHS_*_*', 
        'keep *_patJetsAK8PFCHS_*_*', 
        'keep *_pileupJetIdEvaluator_*_*', 
        'keep *_QGTagger_*_*', 
        'keep *_NjettinessCA8_*_*', 
        'keep *_NjettinessAK8_*_*', 
        'keep *_QJetsAdderCA8_*_*', 
        'keep *_QJetsAdderAK8_*_*', 
        'keep *_ca8PFJetsCHSPrunedMass_*_*', 
        'keep *_ca8PFJetsCHSSoftDropMass_*_*', 
        'keep *_ca8PFJetsCHSTrimmedMass_*_*', 
        'keep *_ca8PFJetsCHSFilteredMass_*_*', 
        'keep *_ak8PFJetsCHSPrunedMass_*_*', 
        'keep *_ak8PFJetsCHSSoftDropMass_*_*', 
        'keep *_ak8PFJetsCHSTrimmedMass_*_*', 
        'keep *_ak8PFJetsCHSFilteredMass_*_*', 
        'keep *_cmsTopTagPFJetsCHSMassCA8_*_*', 
        'keep *_cmsTopTagPFJetsCHSMassAK8_*_*')
)


process.kt4CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.kt4CaloL1FastjetCorrector+process.kt4CaloL2RelativeCorrector+process.kt4CaloL3AbsoluteCorrector+process.kt4CaloResidualCorrector+process.kt4CaloL1FastL2L3ResidualCorrector)


process.pfNoPileUpJMESequence = cms.Sequence(process.goodOfflinePrimaryVertices+process.pfPileUpJME+process.pfNoPileUpJME)


process.ak3PFL2L3CorrectorChain = cms.Sequence(process.ak3PFL2RelativeCorrector+process.ak3PFL3AbsoluteCorrector+process.ak3PFL2L3Corrector)


process.ak4L1JPTOffsetCorrectorChain = cms.Sequence(process.ak4CaloL1OffsetCorrector+process.ak4L1JPTOffsetCorrector)


process.ic5PFL2L3ResidualCorrectorChain = cms.Sequence(process.ic5PFL2RelativeCorrector+process.ic5PFL3AbsoluteCorrector+process.ic5PFResidualCorrector+process.ic5PFL2L3ResidualCorrector)


process.ak2PFL2L3CorrectorChain = cms.Sequence(process.ak2PFL2RelativeCorrector+process.ak2PFL3AbsoluteCorrector+process.ak2PFL2L3Corrector)


process.ak8PFCHSL2L3CorrectorChain = cms.Sequence(process.ak8PFCHSL2RelativeCorrector+process.ak8PFCHSL3AbsoluteCorrector+process.ak8PFCHSL2L3Corrector)


process.patJetFlavourIdLegacy = cms.Sequence(process.patJetPartonsLegacy+process.patJetPartonAssociationLegacy+process.patJetFlavourAssociationLegacy)


process.ak1PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak1PFL1OffsetCorrector+process.ak1PFL2RelativeCorrector+process.ak1PFL3AbsoluteCorrector+process.ak1PFResidualCorrector+process.ak1PFL1L2L3ResidualCorrector)


process.ak4PFCHSL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ak4PFCHSL1FastjetCorrector+process.ak4PFCHSL2RelativeCorrector+process.ak4PFCHSL3AbsoluteCorrector+process.ak4PFCHSResidualCorrector+process.ak4PFCHSL1FastL2L3ResidualCorrector)


process.ak3PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak3PFL2RelativeCorrector+process.ak3PFL3AbsoluteCorrector+process.ak3PFResidualCorrector+process.ak3PFL2L3ResidualCorrector)


process.patJetFlavourId = cms.Sequence(process.patJetPartons+process.patJetFlavourAssociation)


process.ak4CaloL1L2L3CorrectorChain = cms.Sequence(process.ak4CaloL1OffsetCorrector+process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloL1L2L3Corrector)


process.ak9PFCHSL2L3CorrectorChain = cms.Sequence(process.ak9PFCHSL2RelativeCorrector+process.ak9PFCHSL3AbsoluteCorrector+process.ak9PFCHSL2L3Corrector)


process.ak5PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak5PFL1FastjetCorrector+process.ak5PFL2RelativeCorrector+process.ak5PFL3AbsoluteCorrector+process.ak5PFResidualCorrector+process.ak5PFL1FastjetL2L3ResidualCorrector)


process.ak2PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak2PFCHSL2RelativeCorrector+process.ak2PFCHSL3AbsoluteCorrector+process.ak2PFCHSResidualCorrector+process.ak2PFCHSL2L3ResidualCorrector)


process.kt4CaloL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL1L2L3Corrector+process.kt4CaloL6SLBCorrector+process.kt4CaloL1FastL2L3L6Corrector)


process.ic5CaloL1FastL2L3CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.ic5CaloL2RelativeCorrector+process.ic5CaloL3AbsoluteCorrector+process.ic5CaloL1FastL2L3Corrector)


process.ak4PFL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFL1FastL2L3Corrector)


process.ak10PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak10PFCHSL2RelativeCorrector+process.ak10PFCHSL3AbsoluteCorrector+process.ak10PFCHSResidualCorrector+process.ak10PFCHSL2L3ResidualCorrector)


process.ak7PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak7PFL2RelativeCorrector+process.ak7PFL3AbsoluteCorrector+process.ak7PFResidualCorrector+process.ak7PFL2L3ResidualCorrector)


process.ak3PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak3PFL1OffsetCorrector+process.ak3PFL2RelativeCorrector+process.ak3PFL3AbsoluteCorrector+process.ak3PFResidualCorrector+process.ak3PFL1L2L3ResidualCorrector)


process.ak9PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak9PFCHSL1FastjetCorrector+process.ak9PFCHSL2RelativeCorrector+process.ak9PFCHSL3AbsoluteCorrector+process.ak9PFCHSResidualCorrector+process.ak9PFCHSL1FastjetL2L3ResidualCorrector)


process.ak10PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak10PFCHSL1FastjetCorrector+process.ak10PFCHSL2RelativeCorrector+process.ak10PFCHSL3AbsoluteCorrector+process.ak10PFCHSResidualCorrector+process.ak10PFCHSL1FastjetL2L3ResidualCorrector)


process.pfSortByTypePFBRECOSequence = cms.Sequence(process.pfAllNeutralHadronsPFBRECO+process.pfAllChargedHadronsPFBRECO+process.pfAllPhotonsPFBRECO+process.pfAllChargedParticlesPFBRECO+process.pfPileUpAllChargedParticlesPFBRECO+process.pfAllNeutralHadronsAndPhotonsPFBRECO)


process.ak2PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak2PFL1FastjetCorrector+process.ak2PFL2RelativeCorrector+process.ak2PFL3AbsoluteCorrector+process.ak2PFResidualCorrector+process.ak2PFL1FastjetL2L3ResidualCorrector)


process.ak2PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak2PFCHSL1OffsetCorrector+process.ak2PFCHSL2RelativeCorrector+process.ak2PFCHSL3AbsoluteCorrector+process.ak2PFCHSResidualCorrector+process.ak2PFCHSL1L2L3ResidualCorrector)


process.ak4JPTL2L3CorrectorChain = cms.Sequence(process.ak4L1JPTOffsetCorrectorChain+process.ak4JPTL2RelativeCorrector+process.ak4JPTL3AbsoluteCorrector+process.ak4JPTL2L3Corrector)


process.ak4PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFResidualCorrector+process.ak4PFL1FastL2L3ResidualCorrector)


process.ak5PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak5PFCHSL2RelativeCorrector+process.ak5PFCHSL3AbsoluteCorrector+process.ak5PFCHSResidualCorrector+process.ak5PFCHSL2L3ResidualCorrector)


process.ak4JPTL1L2L3CorrectorChain = cms.Sequence(process.ak4L1JPTOffsetCorrectorChain+process.ak4JPTL2RelativeCorrector+process.ak4JPTL3AbsoluteCorrector+process.ak4JPTL1L2L3Corrector)


process.ak7CaloL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak7CaloL1OffsetCorrector+process.ak7CaloL2RelativeCorrector+process.ak7CaloL3AbsoluteCorrector+process.ak7CaloResidualCorrector+process.ak7CaloL1L2L3ResidualCorrector)


process.ak1PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak1PFCHSL2RelativeCorrector+process.ak1PFCHSL3AbsoluteCorrector+process.ak1PFCHSResidualCorrector+process.ak1PFCHSL2L3ResidualCorrector)


process.pfNoPileUpIsoPFBRECOSequence = cms.Sequence(process.pfPileUpIsoPFBRECO+process.pfNoPileUpIsoPFBRECO)


process.ak7CaloL2L3ResidualCorrectorChain = cms.Sequence(process.ak7CaloL2RelativeCorrector+process.ak7CaloL3AbsoluteCorrector+process.ak7CaloResidualCorrector+process.ak7CaloL2L3ResidualCorrector)


process.ic5CaloL2L3ResidualCorrectorChain = cms.Sequence(process.ic5CaloL2RelativeCorrector+process.ic5CaloL3AbsoluteCorrector+process.ic5CaloResidualCorrector+process.ic5CaloL2L3ResidualCorrector)


process.recoPFJets = cms.Sequence(process.fixedGridRhoAll+process.fixedGridRhoFastjetAll+process.fixedGridRhoFastjetCentralChargedPileUp+process.fixedGridRhoFastjetCentralNeutral+process.ak4PFJets+process.pfNoPileUpJMESequence+process.ak4PFJetsCHS+process.ak8PFJetsCHS+process.ak8PFJetsCHSConstituents+process.ak8PFJetsCHSSoftDrop+process.ak8PFJetsCHSSoftDropMass+process.cmsTopTagPFJetsCHS)


process.ak4JPTL1FastL2L3CorrectorChain = cms.Sequence(process.ak4JPTL1FastjetCorrector+process.ak4JPTL2RelativeCorrector+process.ak4JPTL3AbsoluteCorrector+process.ak4JPTL1FastL2L3Corrector)


process.kt4PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.kt4PFL1FastjetCorrector+process.kt4PFL2RelativeCorrector+process.kt4PFL3AbsoluteCorrector+process.kt4PFResidualCorrector+process.kt4PFL1FastL2L3ResidualCorrector)


process.kt6PFL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.kt6PFL2RelativeCorrector+process.kt6PFL3AbsoluteCorrector+process.kt6PFL1FastL2L3Corrector)


process.ak4PFCHSL2L3CorrectorChain = cms.Sequence(process.ak4PFCHSL2RelativeCorrector+process.ak4PFCHSL3AbsoluteCorrector+process.ak4PFCHSL2L3Corrector)


process.ic5PFL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.ic5PFL2RelativeCorrector+process.ic5PFL3AbsoluteCorrector+process.ic5PFL1FastL2L3Corrector)


process.selectedPatCandidates = cms.Sequence(process.selectedPatElectrons+process.selectedPatMuons+process.selectedPatTaus+process.selectedPatPhotons+process.selectedPatJets+process.selectedPatCandidateSummary)


process.ak5PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak5PFCHSL1FastjetCorrector+process.ak5PFCHSL2RelativeCorrector+process.ak5PFCHSL3AbsoluteCorrector+process.ak5PFCHSResidualCorrector+process.ak5PFCHSL1FastjetL2L3ResidualCorrector)


process.photonPFIsolationDepositsPFBRECOSequence = cms.Sequence(process.phPFIsoDepositChargedPFBRECO+process.phPFIsoDepositChargedAllPFBRECO+process.phPFIsoDepositGammaPFBRECO+process.phPFIsoDepositNeutralPFBRECO+process.phPFIsoDepositPUPFBRECO)


process.ak4TrackL1FastL2L3CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.ak4TrackL2RelativeCorrector+process.ak4TrackL3AbsoluteCorrector+process.ak4TrackL1FastL2L3Corrector)


process.ak4PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak4PFL1OffsetCorrector+process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFResidualCorrector+process.ak4PFL1L2L3ResidualCorrector)


process.electronPFIsolationDepositsPATSequence = cms.Sequence(process.elPFIsoDepositChargedPAT+process.elPFIsoDepositChargedAllPAT+process.elPFIsoDepositGammaPAT+process.elPFIsoDepositNeutralPAT+process.elPFIsoDepositPUPAT)


process.kt6PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.kt6PFL1OffsetCorrector+process.kt6PFL2RelativeCorrector+process.kt6PFL3AbsoluteCorrector+process.kt6PFResidualCorrector+process.kt6PFL1L2L3ResidualCorrector)


process.ak6PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak6PFL1FastjetCorrector+process.ak6PFL2RelativeCorrector+process.ak6PFL3AbsoluteCorrector+process.ak6PFResidualCorrector+process.ak6PFL1FastjetL2L3ResidualCorrector)


process.ak4PFL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFL6SLBCorrector+process.ak4PFL1FastL2L3L6Corrector)


process.photonPFIsolationDepositsPATSequence = cms.Sequence(process.phPFIsoDepositChargedPAT+process.phPFIsoDepositChargedAllPAT+process.phPFIsoDepositGammaPAT+process.phPFIsoDepositNeutralPAT+process.phPFIsoDepositPUPAT)


process.ak9PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak9PFCHSL1OffsetCorrector+process.ak9PFCHSL2RelativeCorrector+process.ak9PFCHSL3AbsoluteCorrector+process.ak9PFCHSResidualCorrector+process.ak9PFCHSL1L2L3ResidualCorrector)


process.ak8PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak8PFL1OffsetCorrector+process.ak8PFL2RelativeCorrector+process.ak8PFL3AbsoluteCorrector+process.ak8PFResidualCorrector+process.ak8PFL1L2L3ResidualCorrector)


process.patShrinkingConePFTauDiscrimination = cms.Sequence()


process.ak6PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak6PFCHSL2RelativeCorrector+process.ak6PFCHSL3AbsoluteCorrector+process.ak6PFCHSResidualCorrector+process.ak6PFCHSL2L3ResidualCorrector)


process.ak4JPTL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ak4JPTL1FastjetCorrector+process.ak4JPTL2RelativeCorrector+process.ak4JPTL3AbsoluteCorrector+process.ak4JPTResidualCorrector+process.ak4JPTL1FastL2L3ResidualCorrector)


process.ak4PFL2L3L6CorrectorChain = cms.Sequence(process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFL6SLBCorrector+process.ak4PFL2L3L6Corrector)


process.ak7L1JPTOffsetCorrectorChain = cms.Sequence(process.ak7CaloL1OffsetCorrector+process.ak7L1JPTOffsetCorrector)


process.kt4PFL2L3ResidualCorrectorChain = cms.Sequence(process.kt4PFL2RelativeCorrector+process.kt4PFL3AbsoluteCorrector+process.kt4PFResidualCorrector+process.kt4PFL2L3ResidualCorrector)


process.ak6PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak6PFCHSL1FastjetCorrector+process.ak6PFCHSL2RelativeCorrector+process.ak6PFCHSL3AbsoluteCorrector+process.ak6PFCHSResidualCorrector+process.ak6PFCHSL1FastjetL2L3ResidualCorrector)


process.ak7CaloL1FastL2L3CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.ak7CaloL2RelativeCorrector+process.ak7CaloL3AbsoluteCorrector+process.ak7CaloL1FastL2L3Corrector)


process.kt6PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.kt6PFL1FastjetCorrector+process.kt6PFL2RelativeCorrector+process.kt6PFL3AbsoluteCorrector+process.kt6PFResidualCorrector+process.kt6PFL1FastL2L3ResidualCorrector)


process.pfNoPileUpPFBRECOSequence = cms.Sequence(process.pfPileUpPFBRECO+process.pfNoPileUpPFBRECO)


process.ic5CaloL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL2L3Corrector+process.ic5CaloL6SLBCorrector+process.ic5CaloL2L3L6Corrector)


process.ak7PFL2L3CorrectorChain = cms.Sequence(process.ak7PFL2RelativeCorrector+process.ak7PFL3AbsoluteCorrector+process.ak7PFL2L3Corrector)


process.kt4CaloL2L3ResidualCorrectorChain = cms.Sequence(process.kt4CaloL2RelativeCorrector+process.kt4CaloL3AbsoluteCorrector+process.kt4CaloResidualCorrector+process.kt4CaloL2L3ResidualCorrector)


process.patPFTauIsolation = cms.Sequence(process.tauIsoDepositPFCandidates+process.tauIsoDepositPFChargedHadrons+process.tauIsoDepositPFNeutralHadrons+process.tauIsoDepositPFGammas)


process.ic5CaloL1L2L3CorrectorChain = cms.Sequence(process.ic5CaloL1OffsetCorrector+process.ic5CaloL2RelativeCorrector+process.ic5CaloL3AbsoluteCorrector+process.ic5CaloL1L2L3Corrector)


process.kt6CaloL2L3CorrectorChain = cms.Sequence(process.kt6CaloL2RelativeCorrector+process.kt6CaloL3AbsoluteCorrector+process.kt6CaloL2L3Corrector)


process.ak4CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloResidualCorrector+process.ak4CaloL1FastL2L3ResidualCorrector)


process.ak5PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak5PFCHSL1OffsetCorrector+process.ak5PFCHSL2RelativeCorrector+process.ak5PFCHSL3AbsoluteCorrector+process.ak5PFCHSResidualCorrector+process.ak5PFCHSL1L2L3ResidualCorrector)


process.ak4CaloL2L3L6CorrectorChain = cms.Sequence(process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloL6SLBCorrector+process.ak4CaloL2L3L6Corrector)


process.ak7PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak7PFCHSL1OffsetCorrector+process.ak7PFCHSL2RelativeCorrector+process.ak7PFCHSL3AbsoluteCorrector+process.ak7PFCHSResidualCorrector+process.ak7PFCHSL1L2L3ResidualCorrector)


process.kt6CaloL2L3ResidualCorrectorChain = cms.Sequence(process.kt6CaloL2RelativeCorrector+process.kt6CaloL3AbsoluteCorrector+process.kt6CaloResidualCorrector+process.kt6CaloL2L3ResidualCorrector)


process.ak7PFL2L3L6CorrectorChain = cms.Sequence(process.ak7PFL2L3Corrector+process.ak7PFL6SLBCorrector+process.ak7PFL2L3L6Corrector)


process.ak5PFL2L3CorrectorChain = cms.Sequence(process.ak5PFL2RelativeCorrector+process.ak5PFL3AbsoluteCorrector+process.ak5PFL2L3Corrector)


process.kt4PFL1FastL2L3L6CorrectorChain = cms.Sequence(process.kt4PFL1FastL2L3Corrector+process.kt4PFL6SLBCorrector+process.kt4PFL1FastL2L3L6Corrector)


process.ic5PFL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ic5PFL1FastjetCorrector+process.ic5PFL2RelativeCorrector+process.ic5PFL3AbsoluteCorrector+process.ic5PFResidualCorrector+process.ic5PFL1FastL2L3ResidualCorrector)


process.pfPhotonIsolationPATSequence = cms.Sequence(process.photonPFIsolationDepositsPATSequence+process.phPFIsoValueCharged03PFIdPAT+process.phPFIsoValueChargedAll03PFIdPAT+process.phPFIsoValueGamma03PFIdPAT+process.phPFIsoValueNeutral03PFIdPAT+process.phPFIsoValuePU03PFIdPAT+process.phPFIsoValueCharged04PFIdPAT+process.phPFIsoValueChargedAll04PFIdPAT+process.phPFIsoValueGamma04PFIdPAT+process.phPFIsoValueNeutral04PFIdPAT+process.phPFIsoValuePU04PFIdPAT)


process.ic5PFL2L3CorrectorChain = cms.Sequence(process.ic5PFL2RelativeCorrector+process.ic5PFL3AbsoluteCorrector+process.ic5PFL2L3Corrector)


process.kt4CaloL1L2L3ResidualCorrectorChain = cms.Sequence(process.kt4CaloL1OffsetCorrector+process.kt4CaloL2RelativeCorrector+process.kt4CaloL3AbsoluteCorrector+process.kt4CaloResidualCorrector+process.kt4CaloL1L2L3ResidualCorrector)


process.ak6PFL2L3CorrectorChain = cms.Sequence(process.ak6PFL2RelativeCorrector+process.ak6PFL3AbsoluteCorrector+process.ak6PFL2L3Corrector)


process.patFixedConePFTauDiscrimination = cms.Sequence()


process.kt4PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.kt4PFL1OffsetCorrector+process.kt4PFL2RelativeCorrector+process.kt4PFL3AbsoluteCorrector+process.kt4PFResidualCorrector+process.kt4PFL1L2L3ResidualCorrector)


process.pfElectronIsolationPATSequence = cms.Sequence(process.electronPFIsolationDepositsPATSequence+process.elPFIsoValueCharged03PFIdPAT+process.elPFIsoValueChargedAll03PFIdPAT+process.elPFIsoValueGamma03PFIdPAT+process.elPFIsoValueNeutral03PFIdPAT+process.elPFIsoValuePU03PFIdPAT+process.elPFIsoValueCharged04PFIdPAT+process.elPFIsoValueChargedAll04PFIdPAT+process.elPFIsoValueGamma04PFIdPAT+process.elPFIsoValueNeutral04PFIdPAT+process.elPFIsoValuePU04PFIdPAT+process.elPFIsoValueCharged03NoPFIdPAT+process.elPFIsoValueChargedAll03NoPFIdPAT+process.elPFIsoValueGamma03NoPFIdPAT+process.elPFIsoValueNeutral03NoPFIdPAT+process.elPFIsoValuePU03NoPFIdPAT+process.elPFIsoValueCharged04NoPFIdPAT+process.elPFIsoValueChargedAll04NoPFIdPAT+process.elPFIsoValueGamma04NoPFIdPAT+process.elPFIsoValueNeutral04NoPFIdPAT+process.elPFIsoValuePU04NoPFIdPAT)


process.ak9PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak9PFL1FastjetCorrector+process.ak9PFL2RelativeCorrector+process.ak9PFL3AbsoluteCorrector+process.ak9PFResidualCorrector+process.ak9PFL1FastjetL2L3ResidualCorrector)


process.ic5PFL1FastL2L3L6CorrectorChain = cms.Sequence(process.kt4PFL1FastL2L3Corrector+process.ic5PFL6SLBCorrector+process.ic5PFL1FastL2L3L6Corrector)


process.kt6CaloL1FastL2L3CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.kt6CaloL2RelativeCorrector+process.kt6CaloL3AbsoluteCorrector+process.kt6CaloL1FastL2L3Corrector)


process.Chain = cms.Sequence(process.kt4PFL2L3Corrector+process.kt4PFL6SLBCorrector+process.kt4PFL2L3L6Corrector)


process.ak8PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak8PFCHSL2RelativeCorrector+process.ak8PFCHSL3AbsoluteCorrector+process.ak8PFCHSResidualCorrector+process.ak8PFCHSL2L3ResidualCorrector)


process.ak10PFL2L3CorrectorChain = cms.Sequence(process.ak10PFL2RelativeCorrector+process.ak10PFL3AbsoluteCorrector+process.ak10PFL2L3Corrector)


process.ak7JPTL2L3CorrectorChain = cms.Sequence(process.ak7L1JPTOffsetCorrectorChain+process.ak7JPTL2RelativeCorrector+process.ak7JPTL3AbsoluteCorrector+process.ak7JPTL2L3Corrector)


process.ak8PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak8PFCHSL1OffsetCorrector+process.ak8PFCHSL2RelativeCorrector+process.ak8PFCHSL3AbsoluteCorrector+process.ak8PFCHSResidualCorrector+process.ak8PFCHSL1L2L3ResidualCorrector)


process.ak10PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak10PFL1OffsetCorrector+process.ak10PFL2RelativeCorrector+process.ak10PFL3AbsoluteCorrector+process.ak10PFResidualCorrector+process.ak10PFL1L2L3ResidualCorrector)


process.kt6PFL2L3ResidualCorrectorChain = cms.Sequence(process.kt6PFL2RelativeCorrector+process.kt6PFL3AbsoluteCorrector+process.kt6PFResidualCorrector+process.kt6PFL2L3ResidualCorrector)


process.ak4PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak4PFCHSL1OffsetCorrector+process.ak4PFCHSL2RelativeCorrector+process.ak4PFCHSL3AbsoluteCorrector+process.ak4PFCHSResidualCorrector+process.ak4PFCHSL1L2L3ResidualCorrector)


process.ak4CaloL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak4CaloL1OffsetCorrector+process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloResidualCorrector+process.ak4CaloL1L2L3ResidualCorrector)


process.ak7CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ak7CaloL1FastjetCorrector+process.ak7CaloL2RelativeCorrector+process.ak7CaloL3AbsoluteCorrector+process.ak7CaloResidualCorrector+process.ak7CaloL1FastL2L3ResidualCorrector)


process.ak4PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFResidualCorrector+process.ak4PFL2L3ResidualCorrector)


process.ak2PFCHSL2L3CorrectorChain = cms.Sequence(process.ak2PFCHSL2RelativeCorrector+process.ak2PFCHSL3AbsoluteCorrector+process.ak2PFCHSL2L3Corrector)


process.ak6PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak6PFL1OffsetCorrector+process.ak6PFL2RelativeCorrector+process.ak6PFL3AbsoluteCorrector+process.ak6PFResidualCorrector+process.ak6PFL1L2L3ResidualCorrector)


process.ak10PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak10PFL2RelativeCorrector+process.ak10PFL3AbsoluteCorrector+process.ak10PFResidualCorrector+process.ak10PFL2L3ResidualCorrector)


process.correctionTermsPfMetType1Type2 = cms.Sequence(process.pfJetsPtrForMetCorr+process.particleFlowPtrs+process.pfCandsNotInJetsPtrForMetCorr+process.pfCandsNotInJetsForMetCorr+process.pfCandMETcorr+process.ak4PFL1FastL2L3CorrectorChain+process.corrPfMetType1+process.corrPfMetType2)


process.ak3PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak3PFCHSL2RelativeCorrector+process.ak3PFCHSL3AbsoluteCorrector+process.ak3PFCHSResidualCorrector+process.ak3PFCHSL2L3ResidualCorrector)


process.kt6PFL2L3CorrectorChain = cms.Sequence(process.kt6PFL2RelativeCorrector+process.kt6PFL3AbsoluteCorrector+process.kt6PFL2L3Corrector)


process.ak4PFL1L2L3CorrectorChain = cms.Sequence(process.ak4PFL1OffsetCorrector+process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFL1L2L3Corrector)


process.ak4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFCHSL1FastjetCorrector+process.ak4PFCHSL2RelativeCorrector+process.ak4PFCHSL3AbsoluteCorrector+process.ak4PFCHSL1FastL2L3Corrector)


process.ic5CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ic5CaloL1FastjetCorrector+process.ic5CaloL2RelativeCorrector+process.ic5CaloL3AbsoluteCorrector+process.ic5CaloResidualCorrector+process.ic5CaloL1FastL2L3ResidualCorrector)


process.ak9PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak9PFL1OffsetCorrector+process.ak9PFL2RelativeCorrector+process.ak9PFL3AbsoluteCorrector+process.ak9PFResidualCorrector+process.ak9PFL1L2L3ResidualCorrector)


process.ak7PFL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.ak7PFL2RelativeCorrector+process.ak7PFL3AbsoluteCorrector+process.ak7PFL1FastL2L3Corrector)


process.ak5PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak5PFL1OffsetCorrector+process.ak5PFL2RelativeCorrector+process.ak5PFL3AbsoluteCorrector+process.ak5PFResidualCorrector+process.ak5PFL1L2L3ResidualCorrector)


process.ak10PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak10PFL1FastjetCorrector+process.ak10PFL2RelativeCorrector+process.ak10PFL3AbsoluteCorrector+process.ak10PFResidualCorrector+process.ak10PFL1FastjetL2L3ResidualCorrector)


process.pfParticleSelectionPFBRECOSequence = cms.Sequence(process.pfNoPileUpIsoPFBRECOSequence+process.pfNoPileUpPFBRECOSequence+process.pfSortByTypePFBRECOSequence)


process.ak1PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak1PFCHSL1OffsetCorrector+process.ak1PFCHSL2RelativeCorrector+process.ak1PFCHSL3AbsoluteCorrector+process.ak1PFCHSResidualCorrector+process.ak1PFCHSL1L2L3ResidualCorrector)


process.kt6CaloL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.kt6CaloL1FastjetCorrector+process.kt6CaloL2RelativeCorrector+process.kt6CaloL3AbsoluteCorrector+process.kt6CaloResidualCorrector+process.kt6CaloL1FastL2L3ResidualCorrector)


process.recoGenJets = cms.Sequence(process.ak4GenJets+process.ak8GenJets+process.ak4GenJetsNoNu+process.ak8GenJetsNoNu)


process.ak7JPTL1L2L3CorrectorChain = cms.Sequence(process.ak7JPTL1OffsetCorrector+process.ak7L1JPTOffsetCorrectorChain+process.ak7JPTL2RelativeCorrector+process.ak7JPTL3AbsoluteCorrector+process.ak7JPTL1L2L3Corrector)


process.patPFCandidateIsoDepositSelection = cms.Sequence(process.pfNoPileUpIsoPFBRECOSequence+process.pfSortByTypePFBRECOSequence)


process.genJetParticles = cms.Sequence(process.genParticlesForJets+process.genParticlesForJetsNoNu)


process.ak4TrackL2L3CorrectorChain = cms.Sequence(process.ak4TrackL2RelativeCorrector+process.ak4TrackL3AbsoluteCorrector+process.ak4TrackL2L3Corrector)


process.ak2PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak2PFCHSL1FastjetCorrector+process.ak2PFCHSL2RelativeCorrector+process.ak2PFCHSL3AbsoluteCorrector+process.ak2PFCHSResidualCorrector+process.ak2PFCHSL1FastjetL2L3ResidualCorrector)


process.ak3PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak3PFCHSL1OffsetCorrector+process.ak3PFCHSL2RelativeCorrector+process.ak3PFCHSL3AbsoluteCorrector+process.ak3PFCHSResidualCorrector+process.ak3PFCHSL1L2L3ResidualCorrector)


process.ak3PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak3PFL1FastjetCorrector+process.ak3PFL2RelativeCorrector+process.ak3PFL3AbsoluteCorrector+process.ak3PFResidualCorrector+process.ak3PFL1FastjetL2L3ResidualCorrector)


process.electronPFIsolationDepositsPFBRECOSequence = cms.Sequence(process.elPFIsoDepositChargedPFBRECO+process.elPFIsoDepositChargedAllPFBRECO+process.elPFIsoDepositGammaPFBRECO+process.elPFIsoDepositNeutralPFBRECO+process.elPFIsoDepositPUPFBRECO)


process.kt6CaloL1L2L3ResidualCorrectorChain = cms.Sequence(process.kt6CaloL1OffsetCorrector+process.kt6CaloL2RelativeCorrector+process.kt6CaloL3AbsoluteCorrector+process.kt6CaloResidualCorrector+process.kt6CaloL1L2L3ResidualCorrector)


process.updateHPSPFTaus = cms.Sequence()


process.ak7PFCHSL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFCHSL1FastjetCorrector+process.ak7PFCHSL2RelativeCorrector+process.ak7PFCHSL3AbsoluteCorrector+process.ak7PFCHSL1FastL2L3Corrector)


process.ak5PFCHSL2L3CorrectorChain = cms.Sequence(process.ak5PFCHSL2RelativeCorrector+process.ak5PFCHSL3AbsoluteCorrector+process.ak5PFCHSL2L3Corrector)


process.ak6PFCHSL2L3CorrectorChain = cms.Sequence(process.ak6PFCHSL2RelativeCorrector+process.ak6PFCHSL3AbsoluteCorrector+process.ak6PFCHSL2L3Corrector)


process.ak8PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak8PFL2RelativeCorrector+process.ak8PFL3AbsoluteCorrector+process.ak8PFResidualCorrector+process.ak8PFL2L3ResidualCorrector)


process.ak7CaloL2L3CorrectorChain = cms.Sequence(process.ak7CaloL2RelativeCorrector+process.ak7CaloL3AbsoluteCorrector+process.ak7CaloL2L3Corrector)


process.ak4PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak4PFCHSL2RelativeCorrector+process.ak4PFCHSL3AbsoluteCorrector+process.ak4PFCHSResidualCorrector+process.ak4PFCHSL2L3ResidualCorrector)


process.ak1PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak1PFL1FastjetCorrector+process.ak1PFL2RelativeCorrector+process.ak1PFL3AbsoluteCorrector+process.ak1PFResidualCorrector+process.ak1PFL1FastjetL2L3ResidualCorrector)


process.ic5CaloL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL1L2L3Corrector+process.ic5CaloL6SLBCorrector+process.ic5CaloL1FastL2L3L6Corrector)


process.ak3PFCHSL2L3CorrectorChain = cms.Sequence(process.ak3PFCHSL2RelativeCorrector+process.ak3PFCHSL3AbsoluteCorrector+process.ak3PFCHSL2L3Corrector)


process.ak7PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak7PFL1FastjetCorrector+process.ak7PFL2RelativeCorrector+process.ak7PFL3AbsoluteCorrector+process.ak7PFResidualCorrector+process.ak7PFL1FastjetL2L3ResidualCorrector)


process.ak1PFCHSL2L3CorrectorChain = cms.Sequence(process.ak1PFCHSL2RelativeCorrector+process.ak1PFCHSL3AbsoluteCorrector+process.ak1PFCHSL2L3Corrector)


process.ak7CaloL1L2L3CorrectorChain = cms.Sequence(process.ak7CaloL1OffsetCorrector+process.ak7CaloL2RelativeCorrector+process.ak7CaloL3AbsoluteCorrector+process.ak7CaloL1L2L3Corrector)


process.ak7CaloL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL2L3Corrector+process.ak7CaloL6SLBCorrector+process.ak7CaloL2L3L6Corrector)


process.patCaloTauDiscrimination = cms.Sequence()


process.ak7JPTL1FastL2L3ResidualCorrectorChain = cms.Sequence(process.ak7JPTL1FastjetCorrector+process.ak7L1JPTOffsetCorrectorChain+process.ak7JPTL2RelativeCorrector+process.ak7JPTL3AbsoluteCorrector+process.ak7JPTResidualCorrector+process.ak7JPTL1FastL2L3ResidualCorrector)


process.ak8PFL2L3CorrectorChain = cms.Sequence(process.ak8PFL2RelativeCorrector+process.ak8PFL3AbsoluteCorrector+process.ak8PFL2L3Corrector)


process.ak1PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak1PFL2RelativeCorrector+process.ak1PFL3AbsoluteCorrector+process.ak1PFResidualCorrector+process.ak1PFL2L3ResidualCorrector)


process.ic5CaloL2L3CorrectorChain = cms.Sequence(process.ic5CaloL2RelativeCorrector+process.ic5CaloL3AbsoluteCorrector+process.ic5CaloL2L3Corrector)


process.ak2PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak2PFL1OffsetCorrector+process.ak2PFL2RelativeCorrector+process.ak2PFL3AbsoluteCorrector+process.ak2PFResidualCorrector+process.ak2PFL1L2L3ResidualCorrector)


process.ak4CaloL1FastL2L3CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloL1FastL2L3Corrector)


process.kt4CaloL2L3CorrectorChain = cms.Sequence(process.kt4CaloL2RelativeCorrector+process.kt4CaloL3AbsoluteCorrector+process.kt4CaloL2L3Corrector)


process.ak9PFL2L3CorrectorChain = cms.Sequence(process.ak9PFL2RelativeCorrector+process.ak9PFL3AbsoluteCorrector+process.ak9PFL2L3Corrector)


process.ak10PFCHSL2L3CorrectorChain = cms.Sequence(process.ak10PFCHSL2RelativeCorrector+process.ak10PFCHSL3AbsoluteCorrector+process.ak10PFCHSL2L3Corrector)


process.ak4JPTL2L3ResidualCorrectorChain = cms.Sequence(process.ak4L1JPTOffsetCorrectorChain+process.ak4JPTL2RelativeCorrector+process.ak4JPTL3AbsoluteCorrector+process.ak4JPTResidualCorrector+process.ak4JPTL2L3ResidualCorrector)


process.recoAllGenJetsNoMuNoNu = cms.Sequence(process.kt4GenJetsNoMuNoNu+process.kt6GenJetsNoMuNoNu+process.ak4GenJetsNoMuNoNu+process.ak5GenJetsNoMuNoNu+process.ak8GenJetsNoMuNoNu)


process.kt4PFL1FastL2L3CorrectorChain = cms.Sequence(process.ak4PFL1FastjetCorrector+process.kt4PFL2RelativeCorrector+process.kt4PFL3AbsoluteCorrector+process.kt4PFL1FastL2L3Corrector)


process.kt6CaloL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL1L2L3Corrector+process.kt6CaloL6SLBCorrector+process.kt6CaloL1FastL2L3L6Corrector)


process.kt4CaloL1FastL2L3CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.kt4CaloL2RelativeCorrector+process.kt4CaloL3AbsoluteCorrector+process.kt4CaloL1FastL2L3Corrector)


process.ic5PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ic5PFL1OffsetCorrector+process.ic5PFL2RelativeCorrector+process.ic5PFL3AbsoluteCorrector+process.ic5PFResidualCorrector+process.ic5PFL1L2L3ResidualCorrector)


process.recoAllGenJets = cms.Sequence(process.kt4GenJets+process.kt6GenJets+process.ak4GenJets+process.ak5GenJets+process.ak8GenJets)


process.ak3PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak3PFCHSL1FastjetCorrector+process.ak3PFCHSL2RelativeCorrector+process.ak3PFCHSL3AbsoluteCorrector+process.ak3PFCHSResidualCorrector+process.ak3PFCHSL1FastjetL2L3ResidualCorrector)


process.kt6CaloL1L2L3CorrectorChain = cms.Sequence(process.kt6CaloL1OffsetCorrector+process.kt6CaloL2RelativeCorrector+process.kt6CaloL3AbsoluteCorrector+process.kt6CaloL1L2L3Corrector)


process.ak4PFCHSL1L2L3CorrectorChain = cms.Sequence(process.ak4PFCHSL1OffsetCorrector+process.ak4PFCHSL2RelativeCorrector+process.ak4PFCHSL3AbsoluteCorrector+process.ak4PFCHSL1L2L3Corrector)


process.ak10PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak10PFCHSL1OffsetCorrector+process.ak10PFCHSL2RelativeCorrector+process.ak10PFCHSL3AbsoluteCorrector+process.ak10PFCHSResidualCorrector+process.ak10PFCHSL1L2L3ResidualCorrector)


process.ak1PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak1PFCHSL1FastjetCorrector+process.ak1PFCHSL2RelativeCorrector+process.ak1PFCHSL3AbsoluteCorrector+process.ak1PFCHSResidualCorrector+process.ak1PFCHSL1FastjetL2L3ResidualCorrector)


process.ak1PFL2L3CorrectorChain = cms.Sequence(process.ak1PFL2RelativeCorrector+process.ak1PFL3AbsoluteCorrector+process.ak1PFL2L3Corrector)


process.ak8PFL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak8PFL1FastjetCorrector+process.ak8PFL2RelativeCorrector+process.ak8PFL3AbsoluteCorrector+process.ak8PFResidualCorrector+process.ak8PFL1FastjetL2L3ResidualCorrector)


process.ak7JPTL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak7JPTL1OffsetCorrector+process.ak7L1JPTOffsetCorrectorChain+process.ak7JPTL2RelativeCorrector+process.ak7JPTL3AbsoluteCorrector+process.ak7JPTResidualCorrector+process.ak7JPTL1L2L3ResidualCorrector)


process.kt6PFL1L2L3CorrectorChain = cms.Sequence(process.kt6PFL1OffsetCorrector+process.kt6PFL2RelativeCorrector+process.kt6PFL3AbsoluteCorrector+process.kt6PFL1L2L3Corrector)


process.ic5PFL2L3L6CorrectorChain = cms.Sequence(process.kt4PFL2L3Corrector+process.ic5PFL6SLBCorrector+process.ic5PFL2L3L6Corrector)


process.ak4CaloL2L3CorrectorChain = cms.Sequence(process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloL2L3Corrector)


process.ak5PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak5PFL2RelativeCorrector+process.ak5PFL3AbsoluteCorrector+process.ak5PFResidualCorrector+process.ak5PFL2L3ResidualCorrector)


process.ak4JPTL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak4L1JPTOffsetCorrectorChain+process.ak4JPTL2RelativeCorrector+process.ak4JPTL3AbsoluteCorrector+process.ak4JPTResidualCorrector+process.ak4JPTL1L2L3ResidualCorrector)


process.recoAllGenJetsNoNu = cms.Sequence(process.kt4GenJetsNoNu+process.kt6GenJetsNoNu+process.ak4GenJetsNoNu+process.ak5GenJetsNoNu+process.ak8GenJetsNoNu)


process.ak7PFL1L2L3CorrectorChain = cms.Sequence(process.ak7PFL1OffsetCorrector+process.ak7PFL2RelativeCorrector+process.ak7PFL3AbsoluteCorrector+process.ak7PFL1L2L3Corrector)


process.kt6PFL2L3L6CorrectorChain = cms.Sequence(process.kt4PFL2L3Corrector+process.kt6PFL6SLBCorrector+process.kt6PFL2L3L6Corrector)


process.ak7PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak7PFCHSL2RelativeCorrector+process.ak7PFCHSL3AbsoluteCorrector+process.ak7PFCHSResidualCorrector+process.ak7PFCHSL2L3ResidualCorrector)


process.ak7PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak7PFCHSL1FastjetCorrector+process.ak7PFCHSL2RelativeCorrector+process.ak7PFCHSL3AbsoluteCorrector+process.ak7PFCHSResidualCorrector+process.ak7PFCHSL1FastjetL2L3ResidualCorrector)


process.recoPFJetsWithSubstructure = cms.Sequence(process.fixedGridRhoAll+process.fixedGridRhoFastjetAll+process.fixedGridRhoFastjetCentralChargedPileUp+process.fixedGridRhoFastjetCentralNeutral+process.ak4PFJets+process.ak5PFJets+process.ak7PFJets+process.ak8PFJets+process.ca4PFJets+process.ca8PFJets+process.pfNoPileUpJMESequence+process.ak5PFJetsCHS+process.ak5PFJetsCHSPruned+process.ak5PFJetsCHSFiltered+process.ak5PFJetsCHSTrimmed+process.ak5PFJetsCHSSoftDrop+process.ak4PFJetsCHS+process.ak8PFJetsCHS+process.ak8PFJetsCHSPruned+process.ak8PFJetsCHSFiltered+process.ak8PFJetsCHSTrimmed+process.ak8PFJetsCHSSoftDrop+process.ak8PFJetsCHSConstituents+process.ak8PFJetsCHSPrunedMass+process.ak8PFJetsCHSTrimmedMass+process.ak8PFJetsCHSSoftDropMass+process.ak8PFJetsCHSFilteredMass+process.ca8PFJetsCHS+process.ca8PFJetsCHSPruned+process.ca8PFJetsCHSFiltered+process.ca8PFJetsCHSTrimmed+process.ca8PFJetsCHSSoftDrop+process.cmsTopTagPFJetsCHS+process.hepTopTagPFJetsCHS+process.ca15PFJetsCHSMassDropFiltered+process.ca15PFJetsCHSFiltered+process.ak8PFJetsCS+process.ak8PFJetsCSConstituents+process.ak8PFJetsCSPruned+process.ak8PFJetsCSTrimmed+process.ak8PFJetsCSFiltered+process.puppi+process.ak4PFJetsPuppi+process.softKiller+process.ak4PFJetsSK)


process.kt6CaloL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL2L3Corrector+process.kt6CaloL6SLBCorrector+process.kt6CaloL2L3L6Corrector)


process.patJetCorrections = cms.Sequence(process.patJetCorrFactors)


process.kt4CaloL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL2L3Corrector+process.kt4CaloL6SLBCorrector+process.kt4CaloL2L3L6Corrector)


process.hiRecoGenJets = cms.Sequence(process.iterativeCone5HiGenJets+process.kt4HiGenJets+process.kt6HiGenJets+process.ak1HiGenJets+process.ak2HiGenJets+process.ak3HiGenJets+process.ak4HiGenJets+process.ak5HiGenJets+process.ak6HiGenJets+process.ak7HiGenJets)


process.muonPFIsolationDepositsPATSequence = cms.Sequence(process.muPFIsoDepositChargedPAT+process.muPFIsoDepositChargedAllPAT+process.muPFIsoDepositGammaPAT+process.muPFIsoDepositNeutralPAT+process.muPFIsoDepositPUPAT)


process.ak7CaloL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL1L2L3Corrector+process.ak7CaloL6SLBCorrector+process.ak7CaloL1FastL2L3L6Corrector)


process.kt4PFL1L2L3CorrectorChain = cms.Sequence(process.kt4PFL1OffsetCorrector+process.kt4PFL2RelativeCorrector+process.kt4PFL3AbsoluteCorrector+process.kt4PFL1L2L3Corrector)


process.ak7PFL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak7PFL1OffsetCorrector+process.ak7PFL2RelativeCorrector+process.ak7PFL3AbsoluteCorrector+process.ak7PFResidualCorrector+process.ak7PFL1L2L3ResidualCorrector)


process.kt6PFL1FastL2L3L6CorrectorChain = cms.Sequence(process.kt4PFL1FastL2L3Corrector+process.kt6PFL6SLBCorrector+process.kt6PFL1FastL2L3L6Corrector)


process.kt4CaloL1L2L3CorrectorChain = cms.Sequence(process.kt4CaloL1OffsetCorrector+process.kt4CaloL2RelativeCorrector+process.kt4CaloL3AbsoluteCorrector+process.kt4CaloL1L2L3Corrector)


process.ic5PFL1L2L3CorrectorChain = cms.Sequence(process.ic5PFL1OffsetCorrector+process.ic5PFL2RelativeCorrector+process.ic5PFL3AbsoluteCorrector+process.ic5PFL1L2L3Corrector)


process.ak6PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak6PFL2RelativeCorrector+process.ak6PFL3AbsoluteCorrector+process.ak6PFResidualCorrector+process.ak6PFL2L3ResidualCorrector)


process.muonPFIsolationDepositsPFBRECOSequence = cms.Sequence(process.muPFIsoDepositChargedPFBRECO+process.muPFIsoDepositChargedAllPFBRECO+process.muPFIsoDepositGammaPFBRECO+process.muPFIsoDepositNeutralPFBRECO+process.muPFIsoDepositPUPFBRECO)


process.ak6PFCHSL1L2L3ResidualCorrectorChain = cms.Sequence(process.ak6PFCHSL1OffsetCorrector+process.ak6PFCHSL2RelativeCorrector+process.ak6PFCHSL3AbsoluteCorrector+process.ak6PFCHSResidualCorrector+process.ak6PFCHSL1L2L3ResidualCorrector)


process.ak9PFL2L3ResidualCorrectorChain = cms.Sequence(process.ak9PFL2RelativeCorrector+process.ak9PFL3AbsoluteCorrector+process.ak9PFResidualCorrector+process.ak9PFL2L3ResidualCorrector)


process.ak4CaloL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak4CaloL1FastjetCorrector+process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloL6SLBCorrector+process.ak4CaloL1FastL2L3L6Corrector)


process.ak7PFCHSL2L3CorrectorChain = cms.Sequence(process.ak7PFCHSL2RelativeCorrector+process.ak7PFCHSL3AbsoluteCorrector+process.ak7PFCHSL2L3Corrector)


process.ak7PFL1FastL2L3L6CorrectorChain = cms.Sequence(process.ak7CaloL1L2L3Corrector+process.ak7PFL6SLBCorrector+process.ak7PFL1FastL2L3L6Corrector)


process.kt4PFL2L3CorrectorChain = cms.Sequence(process.kt4PFL2RelativeCorrector+process.kt4PFL3AbsoluteCorrector+process.kt4PFL2L3Corrector)


process.recoAllPFJets = cms.Sequence(process.sisCone5PFJets+process.sisCone7PFJets+process.iterativeCone5PFJets+process.kt4PFJets+process.kt6PFJets+process.kt6PFJetsCentralChargedPileUp+process.kt6PFJetsCentralNeutral+process.kt6PFJetsCentralNeutralTight+process.fixedGridRhoAll+process.fixedGridRhoFastjetAll+process.fixedGridRhoFastjetCentralChargedPileUp+process.fixedGridRhoFastjetCentralNeutral+process.iterativeCone5PFJets+process.ak4PFJets+process.ak5PFJets+process.ak7PFJets+process.ak8PFJets+process.gk5PFJets+process.gk7PFJets+process.ca4PFJets+process.ca8PFJets+process.pfNoPileUpJMESequence+process.ak5PFJetsCHS+process.ak5PFJetsCHSPruned+process.ak5PFJetsCHSFiltered+process.ak5PFJetsCHSTrimmed+process.ak5PFJetsCHSSoftDrop+process.ak4PFJetsCHS+process.ak8PFJetsCHS+process.ak8PFJetsCHSPruned+process.ak8PFJetsCHSFiltered+process.ak8PFJetsCHSTrimmed+process.ak8PFJetsCHSSoftDrop+process.ak8PFJetsCHSPrunedMass+process.ak8PFJetsCHSTrimmedMass+process.ak8PFJetsCHSSoftDropMass+process.ak8PFJetsCHSFilteredMass+process.ca8PFJetsCHS+process.ca8PFJetsCHSPruned+process.ca8PFJetsCHSFiltered+process.ca8PFJetsCHSTrimmed+process.ca8PFJetsCHSSoftDrop+process.cmsTopTagPFJetsCHS+process.hepTopTagPFJetsCHS+process.ca15PFJetsCHSMassDropFiltered+process.ca15PFJetsCHSFiltered+process.ak8PFJetsCS+process.ak8PFJetsCSConstituents+process.ak8PFJetsCSPruned+process.ak8PFJetsCSTrimmed+process.ak8PFJetsCSFiltered)


process.ic5CaloL1L2L3ResidualCorrectorChain = cms.Sequence(process.ic5CaloL1OffsetCorrector+process.ic5CaloL2RelativeCorrector+process.ic5CaloL3AbsoluteCorrector+process.ic5CaloResidualCorrector+process.ic5CaloL1L2L3ResidualCorrector)


process.ak8PFCHSL1FastjetL2L3ResidualCorrectorChain = cms.Sequence(process.ak8PFCHSL1FastjetCorrector+process.ak8PFCHSL2RelativeCorrector+process.ak8PFCHSL3AbsoluteCorrector+process.ak8PFCHSResidualCorrector+process.ak8PFCHSL1FastjetL2L3ResidualCorrector)


process.ak4CaloL2L3ResidualCorrectorChain = cms.Sequence(process.ak4CaloL2RelativeCorrector+process.ak4CaloL3AbsoluteCorrector+process.ak4CaloResidualCorrector+process.ak4CaloL2L3ResidualCorrector)


process.ak4PFL2L3CorrectorChain = cms.Sequence(process.ak4PFL2RelativeCorrector+process.ak4PFL3AbsoluteCorrector+process.ak4PFL2L3Corrector)


process.ak9PFCHSL2L3ResidualCorrectorChain = cms.Sequence(process.ak9PFCHSL2RelativeCorrector+process.ak9PFCHSL3AbsoluteCorrector+process.ak9PFCHSResidualCorrector+process.ak9PFCHSL2L3ResidualCorrector)


process.patHPSPFTauDiscrimination = cms.Sequence(process.updateHPSPFTaus)


process.correctionTermsCaloMet = cms.Sequence(process.ak4CaloL2L3CorrectorChain+process.corrCaloMetType1+process.muCaloMetCorr+process.corrCaloMetType2)


process.pfParticleSelectionForIsoSequence = cms.Sequence(process.particleFlowPtrs+process.pfParticleSelectionPFBRECOSequence)


process.patMETCorrections = cms.Sequence(process.correctionTermsCaloMet+process.caloMetT1+process.caloMetT1T2+process.correctionTermsPfMetType1Type2+process.pfMetT1+process.pfMetT1T2)


process.muonPFIsolationPATSequence = cms.Sequence(process.muonPFIsolationDepositsPATSequence+process.muPFIsoValueCharged03PAT+process.muPFMeanDRIsoValueCharged03PAT+process.muPFSumDRIsoValueCharged03PAT+process.muPFIsoValueChargedAll03PAT+process.muPFMeanDRIsoValueChargedAll03PAT+process.muPFSumDRIsoValueChargedAll03PAT+process.muPFIsoValueGamma03PAT+process.muPFMeanDRIsoValueGamma03PAT+process.muPFSumDRIsoValueGamma03PAT+process.muPFIsoValueNeutral03PAT+process.muPFMeanDRIsoValueNeutral03PAT+process.muPFSumDRIsoValueNeutral03PAT+process.muPFIsoValueGammaHighThreshold03PAT+process.muPFMeanDRIsoValueGammaHighThreshold03PAT+process.muPFSumDRIsoValueGammaHighThreshold03PAT+process.muPFIsoValueNeutralHighThreshold03PAT+process.muPFMeanDRIsoValueNeutralHighThreshold03PAT+process.muPFSumDRIsoValueNeutralHighThreshold03PAT+process.muPFIsoValuePU03PAT+process.muPFMeanDRIsoValuePU03PAT+process.muPFSumDRIsoValuePU03PAT+process.muPFIsoValueCharged04PAT+process.muPFMeanDRIsoValueCharged04PAT+process.muPFSumDRIsoValueCharged04PAT+process.muPFIsoValueChargedAll04PAT+process.muPFMeanDRIsoValueChargedAll04PAT+process.muPFSumDRIsoValueChargedAll04PAT+process.muPFIsoValueGamma04PAT+process.muPFMeanDRIsoValueGamma04PAT+process.muPFSumDRIsoValueGamma04PAT+process.muPFIsoValueNeutral04PAT+process.muPFMeanDRIsoValueNeutral04PAT+process.muPFSumDRIsoValueNeutral04PAT+process.muPFIsoValueGammaHighThreshold04PAT+process.muPFMeanDRIsoValueGammaHighThreshold04PAT+process.muPFSumDRIsoValueGammaHighThreshold04PAT+process.muPFIsoValueNeutralHighThreshold04PAT+process.muPFMeanDRIsoValueNeutralHighThreshold04PAT+process.muPFSumDRIsoValueNeutralHighThreshold04PAT+process.muPFIsoValuePU04PAT+process.muPFMeanDRIsoValuePU04PAT+process.muPFSumDRIsoValuePU04PAT)


process.makePatTaus = cms.Sequence(process.patHPSPFTauDiscrimination+process.patPFCandidateIsoDepositSelection+process.patPFTauIsolation+process.tauMatch+process.tauGenJets+process.tauGenJetsSelectorAllHadrons+process.tauGenJetMatch+process.patTaus)


process.makePatJets = cms.Sequence(process.patJetCorrections+process.patJetCharge+process.patJetPartonMatch+process.patJetGenJetMatch+process.patJetFlavourIdLegacy+process.patJetFlavourId+process.patJets)


process.makePatMuons = cms.Sequence(process.pfParticleSelectionForIsoSequence+process.muonPFIsolationPATSequence+process.muonMatch+process.patMuons)


process.makePatElectrons = cms.Sequence(process.pfParticleSelectionForIsoSequence+process.pfElectronIsolationPATSequence+process.electronMatch+process.patElectrons)


process.makePatMETs = cms.Sequence(process.patMETCorrections+process.patMETs)


process.makePatPhotons = cms.Sequence(process.pfParticleSelectionForIsoSequence+process.pfPhotonIsolationPATSequence+process.photonMatch+process.patPhotons)


process.patCandidates = cms.Sequence(process.makePatElectrons+process.makePatMuons+process.makePatTaus+process.makePatPhotons+process.makePatJets+process.patMETCorrections+process.patMETs+process.patCandidateSummary)


process.outpath = cms.EndPath(process.out)


process.MessageLogger = cms.Service("MessageLogger",
    FrameworkJobReport = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            optionalPSet = cms.untracked.bool(True)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        optionalPSet = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary'),
    cerr = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0),
            optionalPSet = cms.untracked.bool(True)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(1)
        ),
        FwkSummary = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(1)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0),
            optionalPSet = cms.untracked.bool(True)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noTimeStamps = cms.untracked.bool(False),
        optionalPSet = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    cerr_stats = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        output = cms.untracked.string('cerr'),
        threshold = cms.untracked.string('WARNING')
    ),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport'),
    infos = cms.untracked.PSet(
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0),
            optionalPSet = cms.untracked.bool(True)
        ),
        optionalPSet = cms.untracked.bool(True),
        placeholder = cms.untracked.bool(True)
    ),
    statistics = cms.untracked.vstring('cerr_stats'),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring(),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    )
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    QJetsAdderAK8 = cms.PSet(
        initialSeed = cms.untracked.uint32(31)
    ),
    QJetsAdderCA8 = cms.PSet(
        initialSeed = cms.untracked.uint32(7)
    )
)


process.CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(True),
    debugV = cms.untracked.bool(False),
    useCentreTIOffsets = cms.bool(False),
    useDDD = cms.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True)
)


process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'CASTOR', 
        'EcalBarrel', 
        'EcalEndcap', 
        'EcalPreshower', 
        'TOWER')
)


process.CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")


process.CaloTowerGeometryFromDBEP = cms.ESProducer("CaloTowerGeometryFromDBEP",
    applyAlignment = cms.bool(False),
    hcalTopologyConstants = cms.PSet(
        maxDepthHB = cms.int32(2),
        maxDepthHE = cms.int32(3),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.CastorDbProducer = cms.ESProducer("CastorDbProducer")


process.CastorGeometryFromDBEP = cms.ESProducer("CastorGeometryFromDBEP",
    applyAlignment = cms.bool(False)
)


process.DTGeometryESModule = cms.ESProducer("DTGeometryESModule",
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(True),
    fromDDD = cms.bool(False)
)


process.EcalBarrelGeometryFromDBEP = cms.ESProducer("EcalBarrelGeometryFromDBEP",
    applyAlignment = cms.bool(True)
)


process.EcalElectronicsMappingBuilder = cms.ESProducer("EcalElectronicsMappingBuilder")


process.EcalEndcapGeometryFromDBEP = cms.ESProducer("EcalEndcapGeometryFromDBEP",
    applyAlignment = cms.bool(True)
)


process.EcalLaserCorrectionService = cms.ESProducer("EcalLaserCorrectionService")


process.EcalPreshowerGeometryFromDBEP = cms.ESProducer("EcalPreshowerGeometryFromDBEP",
    applyAlignment = cms.bool(True)
)


process.EcalTrigTowerConstituentsMapBuilder = cms.ESProducer("EcalTrigTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EndCap_TTMap.txt')
)


process.GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")


process.HcalAlignmentEP = cms.ESProducer("HcalAlignmentEP")


process.HcalGeometryFromDBEP = cms.ESProducer("HcalGeometryFromDBEP",
    applyAlignment = cms.bool(True),
    hcalTopologyConstants = cms.PSet(
        maxDepthHB = cms.int32(2),
        maxDepthHE = cms.int32(3),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.MuonDetLayerGeometryESProducer = cms.ESProducer("MuonDetLayerGeometryESProducer")


process.ParabolicParametrizedMagneticFieldProducer = cms.ESProducer("AutoParametrizedMagneticFieldProducer",
    label = cms.untracked.string('ParabolicMf'),
    valueOverride = cms.int32(-1),
    version = cms.string('Parabolic')
)


process.RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    compatibiltyWith11 = cms.untracked.bool(True),
    useDDD = cms.untracked.bool(False)
)


process.SiStripRecHitMatcherESProducer = cms.ESProducer("SiStripRecHitMatcherESProducer",
    ComponentName = cms.string('StandardMatcher'),
    NSigmaInside = cms.double(3.0),
    PreFilter = cms.bool(False)
)


process.StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentName = cms.string('StripCPEfromTrackAngle'),
    ComponentType = cms.string('StripCPEfromTrackAngle'),
    parameters = cms.PSet(
        mLC_P0 = cms.double(-0.326),
        mLC_P1 = cms.double(0.618),
        mLC_P2 = cms.double(0.3),
        mTEC_P0 = cms.double(-1.885),
        mTEC_P1 = cms.double(0.471),
        mTIB_P0 = cms.double(-0.742),
        mTIB_P1 = cms.double(0.202),
        mTID_P0 = cms.double(-1.427),
        mTID_P1 = cms.double(0.433),
        mTOB_P0 = cms.double(-1.026),
        mTOB_P1 = cms.double(0.253),
        maxChgOneMIP = cms.double(6000.0),
        useLegacyError = cms.bool(False)
    )
)


process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer")


process.TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)


process.VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducerFromDB",
    debugBuilder = cms.untracked.bool(False),
    label = cms.untracked.string(''),
    valueOverride = cms.int32(-1)
)


process.ZdcGeometryFromDBEP = cms.ESProducer("ZdcGeometryFromDBEP",
    applyAlignment = cms.bool(False)
)


process.ak10PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak10PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFCHSL1Fastjet', 
        'ak10PFCHSL2Relative', 
        'ak10PFCHSL3Absolute', 
        'ak10PFCHSResidual')
)


process.ak10PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFCHSL1Offset', 
        'ak10PFCHSL2Relative', 
        'ak10PFCHSL3Absolute', 
        'ak10PFCHSResidual')
)


process.ak10PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak10PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFCHSL2Relative', 
        'ak10PFCHSL3Absolute')
)


process.ak10PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFCHSL2Relative', 
        'ak10PFCHSL3Absolute', 
        'ak10PFCHSResidual')
)


process.ak10PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L2Relative')
)


process.ak10PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L3Absolute')
)


process.ak10PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK10PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak10PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak10PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFL1Fastjet', 
        'ak10PFL2Relative', 
        'ak10PFL3Absolute', 
        'ak10PFResidual')
)


process.ak10PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFL1Offset', 
        'ak10PFL2Relative', 
        'ak10PFL3Absolute', 
        'ak10PFResidual')
)


process.ak10PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak10PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFL2Relative', 
        'ak10PFL3Absolute')
)


process.ak10PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak10PFL2Relative', 
        'ak10PFL3Absolute', 
        'ak10PFResidual')
)


process.ak10PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L2Relative')
)


process.ak10PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L3Absolute')
)


process.ak10PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK10PF'),
    level = cms.string('L2L3Residual')
)


process.ak1PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak1PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFCHSL1Fastjet', 
        'ak1PFCHSL2Relative', 
        'ak1PFCHSL3Absolute', 
        'ak1PFCHSResidual')
)


process.ak1PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFCHSL1Offset', 
        'ak1PFCHSL2Relative', 
        'ak1PFCHSL3Absolute', 
        'ak1PFCHSResidual')
)


process.ak1PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak1PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFCHSL2Relative', 
        'ak1PFCHSL3Absolute')
)


process.ak1PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFCHSL2Relative', 
        'ak1PFCHSL3Absolute', 
        'ak1PFCHSResidual')
)


process.ak1PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L2Relative')
)


process.ak1PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L3Absolute')
)


process.ak1PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK1PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak1PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak1PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFL1Fastjet', 
        'ak1PFL2Relative', 
        'ak1PFL3Absolute', 
        'ak1PFResidual')
)


process.ak1PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFL1Offset', 
        'ak1PFL2Relative', 
        'ak1PFL3Absolute', 
        'ak1PFResidual')
)


process.ak1PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak1PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFL2Relative', 
        'ak1PFL3Absolute')
)


process.ak1PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak1PFL2Relative', 
        'ak1PFL3Absolute', 
        'ak1PFResidual')
)


process.ak1PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L2Relative')
)


process.ak1PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L3Absolute')
)


process.ak1PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK1PF'),
    level = cms.string('L2L3Residual')
)


process.ak2PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak2PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFCHSL1Fastjet', 
        'ak2PFCHSL2Relative', 
        'ak2PFCHSL3Absolute', 
        'ak2PFCHSResidual')
)


process.ak2PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFCHSL1Offset', 
        'ak2PFCHSL2Relative', 
        'ak2PFCHSL3Absolute', 
        'ak2PFCHSResidual')
)


process.ak2PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak2PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFCHSL2Relative', 
        'ak2PFCHSL3Absolute')
)


process.ak2PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFCHSL2Relative', 
        'ak2PFCHSL3Absolute', 
        'ak2PFCHSResidual')
)


process.ak2PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L2Relative')
)


process.ak2PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L3Absolute')
)


process.ak2PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK2PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak2PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak2PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFL1Fastjet', 
        'ak2PFL2Relative', 
        'ak2PFL3Absolute', 
        'ak2PFResidual')
)


process.ak2PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFL1Offset', 
        'ak2PFL2Relative', 
        'ak2PFL3Absolute', 
        'ak2PFResidual')
)


process.ak2PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak2PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFL2Relative', 
        'ak2PFL3Absolute')
)


process.ak2PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak2PFL2Relative', 
        'ak2PFL3Absolute', 
        'ak2PFResidual')
)


process.ak2PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L2Relative')
)


process.ak2PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L3Absolute')
)


process.ak2PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK2PF'),
    level = cms.string('L2L3Residual')
)


process.ak3PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak3PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFCHSL1Fastjet', 
        'ak3PFCHSL2Relative', 
        'ak3PFCHSL3Absolute', 
        'ak3PFCHSResidual')
)


process.ak3PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFCHSL1Offset', 
        'ak3PFCHSL2Relative', 
        'ak3PFCHSL3Absolute', 
        'ak3PFCHSResidual')
)


process.ak3PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak3PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFCHSL2Relative', 
        'ak3PFCHSL3Absolute')
)


process.ak3PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFCHSL2Relative', 
        'ak3PFCHSL3Absolute', 
        'ak3PFCHSResidual')
)


process.ak3PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L2Relative')
)


process.ak3PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L3Absolute')
)


process.ak3PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK3PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak3PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak3PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFL1Fastjet', 
        'ak3PFL2Relative', 
        'ak3PFL3Absolute', 
        'ak3PFResidual')
)


process.ak3PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFL1Offset', 
        'ak3PFL2Relative', 
        'ak3PFL3Absolute', 
        'ak3PFResidual')
)


process.ak3PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak3PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFL2Relative', 
        'ak3PFL3Absolute')
)


process.ak3PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak3PFL2Relative', 
        'ak3PFL3Absolute', 
        'ak3PFResidual')
)


process.ak3PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L2Relative')
)


process.ak3PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L3Absolute')
)


process.ak3PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK3PF'),
    level = cms.string('L2L3Residual')
)


process.ak4CaloL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'ak4CaloL2Relative', 
        'ak4CaloL3Absolute')
)


process.ak4CaloL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'ak4CaloL2Relative', 
        'ak4CaloL3Absolute', 
        'ak4CaloL6SLB')
)


process.ak4CaloL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'ak4CaloL2Relative', 
        'ak4CaloL3Absolute', 
        'ak4CaloResidual')
)


process.ak4CaloL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak4CaloL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Offset', 
        'ak4CaloL2Relative', 
        'ak4CaloL3Absolute')
)


process.ak4CaloL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Offset', 
        'ak4CaloL2Relative', 
        'ak4CaloL3Absolute', 
        'ak4CaloResidual')
)


process.ak4CaloL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak4CaloL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL2Relative', 
        'ak4CaloL3Absolute')
)


process.ak4CaloL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL2Relative', 
        'ak4CaloL3Absolute', 
        'ak4CaloL6SLB')
)


process.ak4CaloL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL2Relative', 
        'ak4CaloL3Absolute', 
        'ak4CaloResidual')
)


process.ak4CaloL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2Relative')
)


process.ak4CaloL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L3Absolute')
)


process.ak4CaloL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak4CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak4CaloJetsSoftMuonTagInfos")
)


process.ak4CaloResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L2L3Residual')
)


process.ak4JPTL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4JPTL1Fastjet', 
        'ak4JPTL2Relative', 
        'ak4JPTL3Absolute')
)


process.ak4JPTL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4JPTL1Fastjet', 
        'ak4JPTL2Relative', 
        'ak4JPTL3Absolute', 
        'ak4JPTResidual')
)


process.ak4JPTL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak4JPTL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4L1JPTOffset', 
        'ak4JPTL2Relative', 
        'ak4JPTL3Absolute')
)


process.ak4JPTL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4L1JPTOffset', 
        'ak4JPTL2Relative', 
        'ak4JPTL3Absolute', 
        'ak4JPTResidual')
)


process.ak4JPTL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak4JPTL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4L1JPTOffset', 
        'ak4JPTL2Relative', 
        'ak4JPTL3Absolute')
)


process.ak4JPTL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4L1JPTOffset', 
        'ak4JPTL2Relative', 
        'ak4JPTL3Absolute', 
        'ak4JPTResidual')
)


process.ak4JPTL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L2Relative')
)


process.ak4JPTL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L3Absolute')
)


process.ak4JPTResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L2L3Residual')
)


process.ak4L1JPTOffset = cms.ESProducer("L1JPTOffsetCorrectionESProducer",
    algorithm = cms.string('AK5JPT'),
    level = cms.string('L1JPTOffset'),
    offsetService = cms.string('ak4CaloL1Offset')
)


process.ak4PFCHSL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL1Fastjet', 
        'ak4PFCHSL2Relative', 
        'ak4PFCHSL3Absolute')
)


process.ak4PFCHSL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL1Fastjet', 
        'ak4PFCHSL2Relative', 
        'ak4PFCHSL3Absolute', 
        'ak4PFCHSResidual')
)


process.ak4PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak4PFCHSL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL1Offset', 
        'ak4PFCHSL2Relative', 
        'ak4PFCHSL3Absolute')
)


process.ak4PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL1Offset', 
        'ak4PFCHSL2Relative', 
        'ak4PFCHSL3Absolute', 
        'ak4PFCHSResidual')
)


process.ak4PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak4PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL2Relative', 
        'ak4PFCHSL3Absolute')
)


process.ak4PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL2Relative', 
        'ak4PFCHSL3Absolute', 
        'ak4PFCHSResidual')
)


process.ak4PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2Relative')
)


process.ak4PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L3Absolute')
)


process.ak4PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak4PFL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ak4PFL2Relative', 
        'ak4PFL3Absolute')
)


process.ak4PFL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ak4PFL2Relative', 
        'ak4PFL3Absolute', 
        'ak4PFL6SLB')
)


process.ak4PFL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ak4PFL2Relative', 
        'ak4PFL3Absolute', 
        'ak4PFResidual')
)


process.ak4PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak4PFL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Offset', 
        'ak4PFL2Relative', 
        'ak4PFL3Absolute')
)


process.ak4PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Offset', 
        'ak4PFL2Relative', 
        'ak4PFL3Absolute', 
        'ak4PFResidual')
)


process.ak4PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak4PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL2Relative', 
        'ak4PFL3Absolute')
)


process.ak4PFL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL2Relative', 
        'ak4PFL3Absolute', 
        'ak4PFL6SLB')
)


process.ak4PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL2Relative', 
        'ak4PFL3Absolute', 
        'ak4PFResidual')
)


process.ak4PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2Relative')
)


process.ak4PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L3Absolute')
)


process.ak4PFL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak4PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak4PFJetsSoftMuonTagInfos")
)


process.ak4PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK4PF'),
    level = cms.string('L2L3Residual')
)


process.ak4TrackL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'ak4TrackL2Relative', 
        'ak4TrackL3Absolute')
)


process.ak4TrackL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4TrackL2Relative', 
        'ak4TrackL3Absolute')
)


process.ak4TrackL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5TRK'),
    level = cms.string('L2Relative')
)


process.ak4TrackL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5TRK'),
    level = cms.string('L3Absolute')
)


process.ak5PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak5PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFCHSL1Fastjet', 
        'ak5PFCHSL2Relative', 
        'ak5PFCHSL3Absolute', 
        'ak5PFCHSResidual')
)


process.ak5PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFCHSL1Offset', 
        'ak5PFCHSL2Relative', 
        'ak5PFCHSL3Absolute', 
        'ak5PFCHSResidual')
)


process.ak5PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak5PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFCHSL2Relative', 
        'ak5PFCHSL3Absolute')
)


process.ak5PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFCHSL2Relative', 
        'ak5PFCHSL3Absolute', 
        'ak5PFCHSResidual')
)


process.ak5PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L2Relative')
)


process.ak5PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L3Absolute')
)


process.ak5PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak5PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak5PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFL1Fastjet', 
        'ak5PFL2Relative', 
        'ak5PFL3Absolute', 
        'ak5PFResidual')
)


process.ak5PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFL1Offset', 
        'ak5PFL2Relative', 
        'ak5PFL3Absolute', 
        'ak5PFResidual')
)


process.ak5PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak5PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFL2Relative', 
        'ak5PFL3Absolute')
)


process.ak5PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak5PFL2Relative', 
        'ak5PFL3Absolute', 
        'ak5PFResidual')
)


process.ak5PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L2Relative')
)


process.ak5PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L3Absolute')
)


process.ak5PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK5PF'),
    level = cms.string('L2L3Residual')
)


process.ak6PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak6PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFCHSL1Fastjet', 
        'ak6PFCHSL2Relative', 
        'ak6PFCHSL3Absolute', 
        'ak6PFCHSResidual')
)


process.ak6PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFCHSL1Offset', 
        'ak6PFCHSL2Relative', 
        'ak6PFCHSL3Absolute', 
        'ak6PFCHSResidual')
)


process.ak6PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak6PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFCHSL2Relative', 
        'ak6PFCHSL3Absolute')
)


process.ak6PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFCHSL2Relative', 
        'ak6PFCHSL3Absolute', 
        'ak6PFCHSResidual')
)


process.ak6PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L2Relative')
)


process.ak6PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L3Absolute')
)


process.ak6PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK6PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak6PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak6PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFL1Fastjet', 
        'ak6PFL2Relative', 
        'ak6PFL3Absolute', 
        'ak6PFResidual')
)


process.ak6PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFL1Offset', 
        'ak6PFL2Relative', 
        'ak6PFL3Absolute', 
        'ak6PFResidual')
)


process.ak6PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak6PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFL2Relative', 
        'ak6PFL3Absolute')
)


process.ak6PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak6PFL2Relative', 
        'ak6PFL3Absolute', 
        'ak6PFResidual')
)


process.ak6PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L2Relative')
)


process.ak6PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L3Absolute')
)


process.ak6PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK6PF'),
    level = cms.string('L2L3Residual')
)


process.ak7CaloL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'ak7CaloL2Relative', 
        'ak7CaloL3Absolute')
)


process.ak7CaloL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL1Offset', 
        'ak7CaloL2Relative', 
        'ak7CaloL3Absolute', 
        'ak7CaloL6SLB')
)


process.ak7CaloL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL1Fastjet', 
        'ak7CaloL2Relative', 
        'ak7CaloL3Absolute', 
        'ak7CaloResidual')
)


process.ak7CaloL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak7CaloL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL1Offset', 
        'ak7CaloL2Relative', 
        'ak7CaloL3Absolute')
)


process.ak7CaloL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL1Offset', 
        'ak7CaloL2Relative', 
        'ak7CaloL3Absolute', 
        'ak7CaloResidual')
)


process.ak7CaloL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak7CaloL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL2Relative', 
        'ak7CaloL3Absolute')
)


process.ak7CaloL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL2Relative', 
        'ak7CaloL3Absolute', 
        'ak7CaloL6SLB')
)


process.ak7CaloL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7CaloL2Relative', 
        'ak7CaloL3Absolute', 
        'ak7CaloResidual')
)


process.ak7CaloL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L2Relative')
)


process.ak7CaloL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L3Absolute')
)


process.ak7CaloL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak7CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak7CaloJetsSoftMuonTagInfos")
)


process.ak7CaloResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7Calo'),
    level = cms.string('L2L3Residual')
)


process.ak7JPTL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7JPTL1Fastjet', 
        'ak7L1JPTOffset', 
        'ak7JPTL2Relative', 
        'ak7JPTL3Absolute', 
        'ak7JPTResidual')
)


process.ak7JPTL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK7JPT'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ak7JPTL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7JPTL1Offset', 
        'ak7L1JPTOffset', 
        'ak7JPTL2Relative', 
        'ak7JPTL3Absolute')
)


process.ak7JPTL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7JPTL1Offset', 
        'ak7L1JPTOffset', 
        'ak7JPTL2Relative', 
        'ak7JPTL3Absolute', 
        'ak7JPTResidual')
)


process.ak7JPTL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK7JPT'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak7JPTL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7L1JPTOffset', 
        'ak7JPTL2Relative', 
        'ak7JPTL3Absolute')
)


process.ak7L1JPTOffset = cms.ESProducer("L1JPTOffsetCorrectionESProducer",
    algorithm = cms.string('AK7JPT'),
    level = cms.string('L1JPTOffset'),
    offsetService = cms.string('ak4CaloL1Offset')
)


process.ak7PFCHSL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFCHSL1Fastjet', 
        'ak7PFCHSL2Relative', 
        'ak7PFCHSL3Absolute')
)


process.ak7PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak7PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFCHSL1Fastjet', 
        'ak7PFCHSL2Relative', 
        'ak7PFCHSL3Absolute', 
        'ak7PFCHSResidual')
)


process.ak7PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFCHSL1Offset', 
        'ak7PFCHSL2Relative', 
        'ak7PFCHSL3Absolute', 
        'ak7PFCHSResidual')
)


process.ak7PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak7PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFCHSL2Relative', 
        'ak7PFCHSL3Absolute')
)


process.ak7PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFCHSL2Relative', 
        'ak7PFCHSL3Absolute', 
        'ak7PFCHSResidual')
)


process.ak7PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L2Relative')
)


process.ak7PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L3Absolute')
)


process.ak7PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak7PFL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ak7PFL2Relative', 
        'ak7PFL3Absolute')
)


process.ak7PFL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ak7PFL2Relative', 
        'ak7PFL3Absolute', 
        'ak7PFL6SLB')
)


process.ak7PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak7PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFL1Fastjet', 
        'ak7PFL2Relative', 
        'ak7PFL3Absolute', 
        'ak7PFResidual')
)


process.ak7PFL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFL1Offset', 
        'ak7PFL2Relative', 
        'ak7PFL3Absolute')
)


process.ak7PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFL1Offset', 
        'ak7PFL2Relative', 
        'ak7PFL3Absolute', 
        'ak7PFResidual')
)


process.ak7PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak7PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFL2Relative', 
        'ak7PFL3Absolute')
)


process.ak7PFL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFL2Relative', 
        'ak7PFL3Absolute', 
        'ak7PFL6SLB')
)


process.ak7PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak7PFL2Relative', 
        'ak7PFL3Absolute', 
        'ak7PFResidual')
)


process.ak7PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L2Relative')
)


process.ak7PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L3Absolute')
)


process.ak7PFL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ak7PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ak7PFJetsSoftMuonTagInfos")
)


process.ak7PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK7PF'),
    level = cms.string('L2L3Residual')
)


process.ak8PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak8PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFCHSL1Fastjet', 
        'ak8PFCHSL2Relative', 
        'ak8PFCHSL3Absolute', 
        'ak8PFCHSResidual')
)


process.ak8PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFCHSL1Offset', 
        'ak8PFCHSL2Relative', 
        'ak8PFCHSL3Absolute', 
        'ak8PFCHSResidual')
)


process.ak8PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak8PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFCHSL2Relative', 
        'ak8PFCHSL3Absolute')
)


process.ak8PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFCHSL2Relative', 
        'ak8PFCHSL3Absolute', 
        'ak8PFCHSResidual')
)


process.ak8PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L2Relative')
)


process.ak8PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L3Absolute')
)


process.ak8PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak8PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak8PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFL1Fastjet', 
        'ak8PFL2Relative', 
        'ak8PFL3Absolute', 
        'ak8PFResidual')
)


process.ak8PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFL1Offset', 
        'ak8PFL2Relative', 
        'ak8PFL3Absolute', 
        'ak8PFResidual')
)


process.ak8PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak8PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFL2Relative', 
        'ak8PFL3Absolute')
)


process.ak8PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak8PFL2Relative', 
        'ak8PFL3Absolute', 
        'ak8PFResidual')
)


process.ak8PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L2Relative')
)


process.ak8PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L3Absolute')
)


process.ak8PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L2L3Residual')
)


process.ak9PFCHSL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak9PFCHSL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFCHSL1Fastjet', 
        'ak9PFCHSL2Relative', 
        'ak9PFCHSL3Absolute', 
        'ak9PFCHSResidual')
)


process.ak9PFCHSL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFCHSL1Offset', 
        'ak9PFCHSL2Relative', 
        'ak9PFCHSL3Absolute', 
        'ak9PFCHSResidual')
)


process.ak9PFCHSL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak9PFCHSL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFCHSL2Relative', 
        'ak9PFCHSL3Absolute')
)


process.ak9PFCHSL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFCHSL2Relative', 
        'ak9PFCHSL3Absolute', 
        'ak9PFCHSResidual')
)


process.ak9PFCHSL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L2Relative')
)


process.ak9PFCHSL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L3Absolute')
)


process.ak9PFCHSResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK9PFchs'),
    level = cms.string('L2L3Residual')
)


process.ak9PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ak9PFL1FastjetL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFL1Fastjet', 
        'ak9PFL2Relative', 
        'ak9PFL3Absolute', 
        'ak9PFResidual')
)


process.ak9PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFL1Offset', 
        'ak9PFL2Relative', 
        'ak9PFL3Absolute', 
        'ak9PFResidual')
)


process.ak9PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ak9PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFL2Relative', 
        'ak9PFL3Absolute')
)


process.ak9PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak9PFL2Relative', 
        'ak9PFL3Absolute', 
        'ak9PFResidual')
)


process.ak9PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L2Relative')
)


process.ak9PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L3Absolute')
)


process.ak9PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('AK9PF'),
    level = cms.string('L2L3Residual')
)


process.candidateBoostedDoubleSecondaryVertexAK8Computer = cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
    R0 = cms.double(0.8),
    beta = cms.double(1.0),
    maxSVDeltaRToJet = cms.double(0.7),
    weightFile = cms.FileInPath('RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT.weights.xml.gz')
)


process.candidateBoostedDoubleSecondaryVertexCA15Computer = cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
    R0 = cms.double(1.5),
    beta = cms.double(1.0),
    maxSVDeltaRToJet = cms.double(1.3),
    weightFile = cms.FileInPath('RecoBTag/SecondaryVertex/data/BoostedDoubleSV_CA15_BDT.weights.xml.gz')
)


process.candidateCombinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
    calibrationRecord = cms.string('CombinedMVA'),
    jetTagComputers = cms.VPSet(cms.PSet(
        discriminator = cms.bool(True),
        jetTagComputer = cms.string('candidateJetProbabilityComputer'),
        variables = cms.bool(False)
    ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('candidateCombinedSecondaryVertexV2Computer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('softPFMuonComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('softPFElectronComputer'),
            variables = cms.bool(False)
        )),
    useCategories = cms.bool(False)
)


process.candidateCombinedSecondaryVertexComputer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidateCombinedSecondaryVertexSoftLeptonComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLepton', 
        'CombinedSVPseudoVertexNoSoftLepton', 
        'CombinedSVNoVertexNoSoftLepton', 
        'CombinedSVRecoVertexSoftMuon', 
        'CombinedSVPseudoVertexSoftMuon', 
        'CombinedSVNoVertexSoftMuon', 
        'CombinedSVRecoVertexSoftElectron', 
        'CombinedSVPseudoVertexSoftElectron', 
        'CombinedSVNoVertexSoftElectron'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidateCombinedSecondaryVertexSoftLeptonCtagLComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonCtagLESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLeptonCtagL', 
        'CombinedSVPseudoVertexNoSoftLeptonCtagL', 
        'CombinedSVNoVertexNoSoftLeptonCtagL', 
        'CombinedSVRecoVertexSoftMuonCtagL', 
        'CombinedSVPseudoVertexSoftMuonCtagL', 
        'CombinedSVNoVertexSoftMuonCtagL', 
        'CombinedSVRecoVertexSoftElectronCtagL', 
        'CombinedSVPseudoVertexSoftElectronCtagL', 
        'CombinedSVNoVertexSoftElectronCtagL'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidateCombinedSecondaryVertexV2Computer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVIVFV2RecoVertex', 
        'CombinedSVIVFV2PseudoVertex', 
        'CombinedSVIVFV2NoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidateJetBProbabilityComputer = cms.ESProducer("CandidateJetBProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.candidateJetProbabilityComputer = cms.ESProducer("CandidateJetProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(0.3),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.candidateNegativeCombinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
    calibrationRecord = cms.string('CombinedMVA'),
    jetTagComputers = cms.VPSet(cms.PSet(
        discriminator = cms.bool(True),
        jetTagComputer = cms.string('candidateNegativeOnlyJetProbabilityComputer'),
        variables = cms.bool(False)
    ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('candidateNegativeCombinedSecondaryVertexComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('negativeSoftPFMuonComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('negativeSoftPFElectronComputer'),
            variables = cms.bool(False)
        )),
    useCategories = cms.bool(False)
)


process.candidateNegativeCombinedSecondaryVertexComputer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(True),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(-2.0),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(True)
)


process.candidateNegativeCombinedSecondaryVertexSoftLeptonComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLepton', 
        'CombinedSVPseudoVertexNoSoftLepton', 
        'CombinedSVNoVertexNoSoftLepton', 
        'CombinedSVRecoVertexSoftMuon', 
        'CombinedSVPseudoVertexSoftMuon', 
        'CombinedSVNoVertexSoftMuon', 
        'CombinedSVRecoVertexSoftElectron', 
        'CombinedSVPseudoVertexSoftElectron', 
        'CombinedSVNoVertexSoftElectron'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(True),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(-2.0),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(True)
)


process.candidateNegativeCombinedSecondaryVertexSoftLeptonCtagLComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonCtagLESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLeptonCtagL', 
        'CombinedSVPseudoVertexNoSoftLeptonCtagL', 
        'CombinedSVNoVertexNoSoftLeptonCtagL', 
        'CombinedSVRecoVertexSoftMuonCtagL', 
        'CombinedSVPseudoVertexSoftMuonCtagL', 
        'CombinedSVNoVertexSoftMuonCtagL', 
        'CombinedSVRecoVertexSoftElectronCtagL', 
        'CombinedSVPseudoVertexSoftElectronCtagL', 
        'CombinedSVNoVertexSoftElectronCtagL'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(True),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(-2.0),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(True)
)


process.candidateNegativeCombinedSecondaryVertexV2Computer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVIVFV2RecoVertex', 
        'CombinedSVIVFV2PseudoVertex', 
        'CombinedSVIVFV2NoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(True),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(-2.0),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(True)
)


process.candidateNegativeOnlyJetBProbabilityComputer = cms.ESProducer("CandidateJetBProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    trackIpSign = cms.int32(-1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.candidateNegativeOnlyJetProbabilityComputer = cms.ESProducer("CandidateJetProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(0.3),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    trackIpSign = cms.int32(-1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.candidateNegativeTrackCounting3D2ndComputer = cms.ESProducer("CandidateNegativeTrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(2),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.candidateNegativeTrackCounting3D3rdComputer = cms.ESProducer("CandidateNegativeTrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(3),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.candidatePositiveCombinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
    calibrationRecord = cms.string('CombinedMVA'),
    jetTagComputers = cms.VPSet(cms.PSet(
        discriminator = cms.bool(True),
        jetTagComputer = cms.string('candidatePositiveOnlyJetProbabilityComputer'),
        variables = cms.bool(False)
    ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('candidatePositiveCombinedSecondaryVertexComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('positiveSoftPFMuonComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('positiveSoftPFElectronComputer'),
            variables = cms.bool(False)
        )),
    useCategories = cms.bool(False)
)


process.candidatePositiveCombinedSecondaryVertexComputer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidatePositiveCombinedSecondaryVertexSoftLeptonComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLepton', 
        'CombinedSVPseudoVertexNoSoftLepton', 
        'CombinedSVNoVertexNoSoftLepton', 
        'CombinedSVRecoVertexSoftMuon', 
        'CombinedSVPseudoVertexSoftMuon', 
        'CombinedSVNoVertexSoftMuon', 
        'CombinedSVRecoVertexSoftElectron', 
        'CombinedSVPseudoVertexSoftElectron', 
        'CombinedSVNoVertexSoftElectron'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidatePositiveCombinedSecondaryVertexSoftLeptonCtagLComputer = cms.ESProducer("CandidateCombinedSecondaryVertexSoftLeptonCtagLESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLeptonCtagL', 
        'CombinedSVPseudoVertexNoSoftLeptonCtagL', 
        'CombinedSVNoVertexNoSoftLeptonCtagL', 
        'CombinedSVRecoVertexSoftMuonCtagL', 
        'CombinedSVPseudoVertexSoftMuonCtagL', 
        'CombinedSVNoVertexSoftMuonCtagL', 
        'CombinedSVRecoVertexSoftElectronCtagL', 
        'CombinedSVPseudoVertexSoftElectronCtagL', 
        'CombinedSVNoVertexSoftElectronCtagL'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidatePositiveCombinedSecondaryVertexV2Computer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVIVFV2RecoVertex', 
        'CombinedSVIVFV2PseudoVertex', 
        'CombinedSVIVFV2NoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.candidatePositiveOnlyJetBProbabilityComputer = cms.ESProducer("CandidateJetBProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.candidatePositiveOnlyJetProbabilityComputer = cms.ESProducer("CandidateJetProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(0.3),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.candidateSimpleSecondaryVertex2TrkComputer = cms.ESProducer("CandidateSimpleSecondaryVertexESProducer",
    minTracks = cms.uint32(2),
    unBoost = cms.bool(False),
    use3d = cms.bool(True),
    useSignificance = cms.bool(True)
)


process.candidateSimpleSecondaryVertex3TrkComputer = cms.ESProducer("CandidateSimpleSecondaryVertexESProducer",
    minTracks = cms.uint32(3),
    unBoost = cms.bool(False),
    use3d = cms.bool(True),
    useSignificance = cms.bool(True)
)


process.candidateTrackCounting3D2ndComputer = cms.ESProducer("CandidateTrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(2),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.candidateTrackCounting3D3rdComputer = cms.ESProducer("CandidateTrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(3),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.combinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
    calibrationRecord = cms.string('CombinedMVA'),
    jetTagComputers = cms.VPSet(cms.PSet(
        discriminator = cms.bool(True),
        jetTagComputer = cms.string('jetProbabilityComputer'),
        variables = cms.bool(False)
    ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('combinedSecondaryVertexV2Computer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('softPFMuonComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('softPFElectronComputer'),
            variables = cms.bool(False)
        )),
    useCategories = cms.bool(False)
)


process.combinedSecondaryVertexComputer = cms.ESProducer("CombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.combinedSecondaryVertexSoftLeptonComputer = cms.ESProducer("CombinedSecondaryVertexSoftLeptonESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertexNoSoftLepton', 
        'CombinedSVPseudoVertexNoSoftLepton', 
        'CombinedSVNoVertexNoSoftLepton', 
        'CombinedSVRecoVertexSoftMuon', 
        'CombinedSVPseudoVertexSoftMuon', 
        'CombinedSVNoVertexSoftMuon', 
        'CombinedSVRecoVertexSoftElectron', 
        'CombinedSVPseudoVertexSoftElectron', 
        'CombinedSVNoVertexSoftElectron'),
    categoryVariableName = cms.string('vertexLeptonCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.combinedSecondaryVertexV2Computer = cms.ESProducer("CombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVIVFV2RecoVertex', 
        'CombinedSVIVFV2PseudoVertex', 
        'CombinedSVIVFV2NoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.doubleVertex2TrkComputer = cms.ESProducer("SimpleSecondaryVertexESProducer",
    minTracks = cms.uint32(2),
    minVertices = cms.uint32(2),
    unBoost = cms.bool(False),
    use3d = cms.bool(True),
    useSignificance = cms.bool(True)
)


process.fakeForIdealAlignment = cms.ESProducer("FakeAlignmentProducer",
    appendToDataLabel = cms.string('fakeForIdeal')
)


process.ghostTrackComputer = cms.ESProducer("GhostTrackESProducer",
    calibrationRecords = cms.vstring('GhostTrackRecoVertex', 
        'GhostTrackPseudoVertex', 
        'GhostTrackNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    minimumTrackWeight = cms.double(0.5),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True)
)


process.hcalTopologyIdeal = cms.ESProducer("HcalTopologyIdealEP",
    Exclude = cms.untracked.string(''),
    appendToDataLabel = cms.string(''),
    hcalTopologyConstants = cms.PSet(
        maxDepthHB = cms.int32(2),
        maxDepthHE = cms.int32(3),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)


process.ic5CaloL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'ic5CaloL2Relative', 
        'ic5CaloL3Absolute')
)


process.ic5CaloL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL1Offset', 
        'ic5CaloL2Relative', 
        'ic5CaloL3Absolute', 
        'ic5CaloL6SLB')
)


process.ic5CaloL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL1Fastjet', 
        'ic5CaloL2Relative', 
        'ic5CaloL3Absolute', 
        'ic5CaloResidual')
)


process.ic5CaloL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.ic5CaloL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL1Offset', 
        'ic5CaloL2Relative', 
        'ic5CaloL3Absolute')
)


process.ic5CaloL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL1Offset', 
        'ic5CaloL2Relative', 
        'ic5CaloL3Absolute', 
        'ic5CaloResidual')
)


process.ic5CaloL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ic5CaloL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL2Relative', 
        'ic5CaloL3Absolute')
)


process.ic5CaloL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL2Relative', 
        'ic5CaloL3Absolute', 
        'ic5CaloL6SLB')
)


process.ic5CaloL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5CaloL2Relative', 
        'ic5CaloL3Absolute', 
        'ic5CaloResidual')
)


process.ic5CaloL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L2Relative')
)


process.ic5CaloL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L3Absolute')
)


process.ic5CaloL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ic5CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ic5CaloJetsSoftMuonTagInfos")
)


process.ic5CaloResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('IC5Calo'),
    level = cms.string('L2L3Residual')
)


process.ic5PFL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ic5PFL2Relative', 
        'ic5PFL3Absolute')
)


process.ic5PFL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'ic5PFL2Relative', 
        'ic5PFL3Absolute', 
        'ic5PFL6SLB')
)


process.ic5PFL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5PFL1Fastjet', 
        'ic5PFL2Relative', 
        'ic5PFL3Absolute', 
        'ic5PFResidual')
)


process.ic5PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.ic5PFL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5PFL1Offset', 
        'ic5PFL2Relative', 
        'ic5PFL3Absolute')
)


process.ic5PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5PFL1Offset', 
        'ic5PFL2Relative', 
        'ic5PFL3Absolute', 
        'ic5PFResidual')
)


process.ic5PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.ic5PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5PFL2Relative', 
        'ic5PFL3Absolute')
)


process.ic5PFL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5PFL2Relative', 
        'ic5PFL3Absolute', 
        'ic5PFL6SLB')
)


process.ic5PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ic5PFL2Relative', 
        'ic5PFL3Absolute', 
        'ic5PFResidual')
)


process.ic5PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L2Relative')
)


process.ic5PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L3Absolute')
)


process.ic5PFL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("ic5PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("ic5PFJetsSoftMuonTagInfos")
)


process.ic5PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('IC5PF'),
    level = cms.string('L2L3Residual')
)


process.idealForDigiCSCGeometry = cms.ESProducer("CSCGeometryESModule",
    alignmentsLabel = cms.string('fakeForIdeal'),
    appendToDataLabel = cms.string('idealForDigi'),
    applyAlignment = cms.bool(False),
    debugV = cms.untracked.bool(False),
    useCentreTIOffsets = cms.bool(False),
    useDDD = cms.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True)
)


process.idealForDigiDTGeometry = cms.ESProducer("DTGeometryESModule",
    alignmentsLabel = cms.string('fakeForIdeal'),
    appendToDataLabel = cms.string('idealForDigi'),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(False)
)


process.idealForDigiTrackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    alignmentsLabel = cms.string('fakeForIdeal'),
    appendToDataLabel = cms.string('idealForDigi'),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(False)
)


process.impactParameterMVAComputer = cms.ESProducer("GenericMVAJetTagESProducer",
    calibrationRecord = cms.string('ImpactParameterMVA'),
    useCategories = cms.bool(False)
)


process.jetBProbabilityComputer = cms.ESProducer("JetBProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.jetProbabilityComputer = cms.ESProducer("JetProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(0.3),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.kt4CaloL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'kt4CaloL2Relative', 
        'kt4CaloL3Absolute')
)


process.kt4CaloL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL1Offset', 
        'kt4CaloL2Relative', 
        'kt4CaloL3Absolute', 
        'kt4CaloL6SLB')
)


process.kt4CaloL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL1Fastjet', 
        'kt4CaloL2Relative', 
        'kt4CaloL3Absolute', 
        'kt4CaloResidual')
)


process.kt4CaloL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.kt4CaloL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL1Offset', 
        'kt4CaloL2Relative', 
        'kt4CaloL3Absolute')
)


process.kt4CaloL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL1Offset', 
        'kt4CaloL2Relative', 
        'kt4CaloL3Absolute', 
        'kt4CaloResidual')
)


process.kt4CaloL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.kt4CaloL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL2Relative', 
        'kt4CaloL3Absolute')
)


process.kt4CaloL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL2Relative', 
        'kt4CaloL3Absolute', 
        'kt4CaloL6SLB')
)


process.kt4CaloL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4CaloL2Relative', 
        'kt4CaloL3Absolute', 
        'kt4CaloResidual')
)


process.kt4CaloL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L2Relative')
)


process.kt4CaloL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L3Absolute')
)


process.kt4CaloL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt4CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt4CaloJetsSoftMuonTagInfos")
)


process.kt4CaloResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT4Calo'),
    level = cms.string('L2L3Residual')
)


process.kt4PFL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'kt4PFL2Relative', 
        'kt4PFL3Absolute')
)


process.kt4PFL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'kt4PFL2Relative', 
        'kt4PFL3Absolute', 
        'kt4PFL6SLB')
)


process.kt4PFL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4PFL1Fastjet', 
        'kt4PFL2Relative', 
        'kt4PFL3Absolute', 
        'kt4PFResidual')
)


process.kt4PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.kt4PFL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4PFL1Offset', 
        'kt4PFL2Relative', 
        'kt4PFL3Absolute')
)


process.kt4PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4PFL1Offset', 
        'kt4PFL2Relative', 
        'kt4PFL3Absolute', 
        'kt4PFResidual')
)


process.kt4PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.kt4PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4PFL2Relative', 
        'kt4PFL3Absolute')
)


process.kt4PFL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4PFL2Relative', 
        'kt4PFL3Absolute', 
        'kt4PFL6SLB')
)


process.kt4PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt4PFL2Relative', 
        'kt4PFL3Absolute', 
        'kt4PFResidual')
)


process.kt4PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L2Relative')
)


process.kt4PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L3Absolute')
)


process.kt4PFL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt4PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt4PFJetsSoftMuonTagInfos")
)


process.kt4PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT4PF'),
    level = cms.string('L2L3Residual')
)


process.kt6CaloL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4CaloL1Fastjet', 
        'kt6CaloL2Relative', 
        'kt6CaloL3Absolute')
)


process.kt6CaloL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL1Offset', 
        'kt6CaloL2Relative', 
        'kt6CaloL3Absolute', 
        'kt6CaloL6SLB')
)


process.kt6CaloL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL1Fastjet', 
        'kt6CaloL2Relative', 
        'kt6CaloL3Absolute', 
        'kt6CaloResidual')
)


process.kt6CaloL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAllCalo")
)


process.kt6CaloL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL1Offset', 
        'kt6CaloL2Relative', 
        'kt6CaloL3Absolute')
)


process.kt6CaloL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL1Offset', 
        'kt6CaloL2Relative', 
        'kt6CaloL3Absolute', 
        'kt6CaloResidual')
)


process.kt6CaloL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.kt6CaloL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL2Relative', 
        'kt6CaloL3Absolute')
)


process.kt6CaloL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL2Relative', 
        'kt6CaloL3Absolute', 
        'kt6CaloL6SLB')
)


process.kt6CaloL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6CaloL2Relative', 
        'kt6CaloL3Absolute', 
        'kt6CaloResidual')
)


process.kt6CaloL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L2Relative')
)


process.kt6CaloL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L3Absolute')
)


process.kt6CaloL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(True),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt6CaloJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt6CaloJetsSoftMuonTagInfos")
)


process.kt6CaloResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT6Calo'),
    level = cms.string('L2L3Residual')
)


process.kt6PFL1FastL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'kt6PFL2Relative', 
        'kt6PFL3Absolute')
)


process.kt6PFL1FastL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('ak4PFL1Fastjet', 
        'kt6PFL2Relative', 
        'kt6PFL3Absolute', 
        'kt6PFL6SLB')
)


process.kt6PFL1FastL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6PFL1Fastjet', 
        'kt6PFL2Relative', 
        'kt6PFL3Absolute', 
        'kt6PFResidual')
)


process.kt6PFL1Fastjet = cms.ESProducer("L1FastjetCorrectionESProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L1FastJet'),
    srcRho = cms.InputTag("fixedGridRhoFastjetAll")
)


process.kt6PFL1L2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6PFL1Offset', 
        'kt6PFL2Relative', 
        'kt6PFL3Absolute')
)


process.kt6PFL1L2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6PFL1Offset', 
        'kt6PFL2Relative', 
        'kt6PFL3Absolute', 
        'kt6PFResidual')
)


process.kt6PFL1Offset = cms.ESProducer("L1OffsetCorrectionESProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L1Offset'),
    minVtxNdof = cms.int32(4),
    vertexCollection = cms.string('offlinePrimaryVertices')
)


process.kt6PFL2L3 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6PFL2Relative', 
        'kt6PFL3Absolute')
)


process.kt6PFL2L3L6 = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6PFL2Relative', 
        'kt6PFL3Absolute', 
        'kt6PFL6SLB')
)


process.kt6PFL2L3Residual = cms.ESProducer("JetCorrectionESChain",
    correctors = cms.vstring('kt6PFL2Relative', 
        'kt6PFL3Absolute', 
        'kt6PFResidual')
)


process.kt6PFL2Relative = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L2Relative')
)


process.kt6PFL3Absolute = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L3Absolute')
)


process.kt6PFL6SLB = cms.ESProducer("L6SLBCorrectionESProducer",
    addMuonToJet = cms.bool(False),
    algorithm = cms.string(''),
    level = cms.string('L6SLB'),
    srcBTagInfoElectron = cms.InputTag("kt6PFJetsSoftElectronTagInfos"),
    srcBTagInfoMuon = cms.InputTag("kt6PFJetsSoftMuonTagInfos")
)


process.kt6PFResidual = cms.ESProducer("LXXXCorrectionESProducer",
    algorithm = cms.string('KT6PF'),
    level = cms.string('L2L3Residual')
)


process.negativeCombinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
    calibrationRecord = cms.string('CombinedMVA'),
    jetTagComputers = cms.VPSet(cms.PSet(
        discriminator = cms.bool(True),
        jetTagComputer = cms.string('negativeOnlyJetProbabilityComputer'),
        variables = cms.bool(False)
    ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('negativeCombinedSecondaryVertexComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('negativeSoftPFMuonComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('negativeSoftPFElectronComputer'),
            variables = cms.bool(False)
        )),
    useCategories = cms.bool(False)
)


process.negativeCombinedSecondaryVertexComputer = cms.ESProducer("CombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(True),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(-2.0),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(True)
)


process.negativeCombinedSecondaryVertexV2Computer = cms.ESProducer("CombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVIVFV2RecoVertex', 
        'CombinedSVIVFV2PseudoVertex', 
        'CombinedSVIVFV2NoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(True),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(-2.0),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(True)
)


process.negativeOnlyJetBProbabilityComputer = cms.ESProducer("JetBProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    trackIpSign = cms.int32(-1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.negativeOnlyJetProbabilityComputer = cms.ESProducer("JetProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(0.3),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    trackIpSign = cms.int32(-1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.negativeSoftPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('negative'),
    use3d = cms.bool(False)
)


process.negativeSoftPFElectronByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('negative'),
    use3d = cms.bool(True)
)


process.negativeSoftPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('negative')
)


process.negativeSoftPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string('negative'),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz')
)


process.negativeSoftPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('negative'),
    use3d = cms.bool(False)
)


process.negativeSoftPFMuonByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('negative'),
    use3d = cms.bool(True)
)


process.negativeSoftPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('negative')
)


process.negativeSoftPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string('negative'),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz')
)


process.negativeTrackCounting3D2ndComputer = cms.ESProducer("NegativeTrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(2),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.negativeTrackCounting3D3rdComputer = cms.ESProducer("NegativeTrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(3),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.positiveCombinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
    calibrationRecord = cms.string('CombinedMVA'),
    jetTagComputers = cms.VPSet(cms.PSet(
        discriminator = cms.bool(True),
        jetTagComputer = cms.string('positiveOnlyJetProbabilityComputer'),
        variables = cms.bool(False)
    ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('positiveCombinedSecondaryVertexComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('positiveSoftPFMuonComputer'),
            variables = cms.bool(False)
        ), 
        cms.PSet(
            discriminator = cms.bool(True),
            jetTagComputer = cms.string('positiveSoftPFElectronComputer'),
            variables = cms.bool(False)
        )),
    useCategories = cms.bool(False)
)


process.positiveCombinedSecondaryVertexComputer = cms.ESProducer("CombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.positiveCombinedSecondaryVertexV2Computer = cms.ESProducer("CombinedSecondaryVertexESProducer",
    calibrationRecords = cms.vstring('CombinedSVIVFV2RecoVertex', 
        'CombinedSVIVFV2PseudoVertex', 
        'CombinedSVIVFV2NoVertex'),
    categoryVariableName = cms.string('vertexCategory'),
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(0),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useCategories = cms.bool(True),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)


process.positiveOnlyJetBProbabilityComputer = cms.ESProducer("JetBProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.positiveOnlyJetProbabilityComputer = cms.ESProducer("JetProbabilityESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(0.3),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumProbability = cms.double(0.005),
    trackIpSign = cms.int32(1),
    trackQualityClass = cms.string('any'),
    useVariableJTA = cms.bool(False)
)


process.positiveSoftPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('positive'),
    use3d = cms.bool(False)
)


process.positiveSoftPFElectronByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('positive'),
    use3d = cms.bool(True)
)


process.positiveSoftPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('positive')
)


process.positiveSoftPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string('positive'),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz')
)


process.positiveSoftPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('positive'),
    use3d = cms.bool(False)
)


process.positiveSoftPFMuonByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('positive'),
    use3d = cms.bool(True)
)


process.positiveSoftPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('positive')
)


process.positiveSoftPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string('positive'),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz')
)


process.siPixelQualityESProducer = cms.ESProducer("SiPixelQualityESProducer",
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelQualityFromDbRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiPixelDetVOffRcd'),
            tag = cms.string('')
        ))
)


process.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer("SiStripBackPlaneCorrectionDepESProducer",
    BackPlaneCorrectionDeconvMode = cms.PSet(
        label = cms.untracked.string('deconvolution'),
        record = cms.string('SiStripBackPlaneCorrectionRcd')
    ),
    BackPlaneCorrectionPeakMode = cms.PSet(
        label = cms.untracked.string('peak'),
        record = cms.string('SiStripBackPlaneCorrectionRcd')
    ),
    LatencyRecord = cms.PSet(
        label = cms.untracked.string(''),
        record = cms.string('SiStripLatencyRcd')
    )
)


process.siStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    APVGain = cms.VPSet(cms.PSet(
        Label = cms.untracked.string(''),
        NormalizationFactor = cms.untracked.double(1.0),
        Record = cms.string('SiStripApvGainRcd')
    ), 
        cms.PSet(
            Label = cms.untracked.string(''),
            NormalizationFactor = cms.untracked.double(1.0),
            Record = cms.string('SiStripApvGain2Rcd')
        )),
    AutomaticNormalization = cms.bool(False),
    appendToDataLabel = cms.string(''),
    printDebug = cms.untracked.bool(False)
)


process.siStripLorentzAngleDepESProducer = cms.ESProducer("SiStripLorentzAngleDepESProducer",
    LatencyRecord = cms.PSet(
        label = cms.untracked.string(''),
        record = cms.string('SiStripLatencyRcd')
    ),
    LorentzAngleDeconvMode = cms.PSet(
        label = cms.untracked.string('deconvolution'),
        record = cms.string('SiStripLorentzAngleRcd')
    ),
    LorentzAnglePeakMode = cms.PSet(
        label = cms.untracked.string('peak'),
        record = cms.string('SiStripLorentzAngleRcd')
    )
)


process.siStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiStripDetCablingRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('RunInfoRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadFiberRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadModuleRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadStripRcd'),
            tag = cms.string('')
        )),
    PrintDebugOutput = cms.bool(False),
    ReduceGranularity = cms.bool(False),
    ThresholdForReducedGranularity = cms.double(0.3),
    UseEmptyRunInfo = cms.bool(False),
    appendToDataLabel = cms.string('')
)


process.simpleSecondaryVertex2TrkComputer = cms.ESProducer("SimpleSecondaryVertexESProducer",
    minTracks = cms.uint32(2),
    unBoost = cms.bool(False),
    use3d = cms.bool(True),
    useSignificance = cms.bool(True)
)


process.simpleSecondaryVertex3TrkComputer = cms.ESProducer("SimpleSecondaryVertexESProducer",
    minTracks = cms.uint32(3),
    unBoost = cms.bool(False),
    use3d = cms.bool(True),
    useSignificance = cms.bool(True)
)


process.sistripconn = cms.ESProducer("SiStripConnectivity")


process.softPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('any'),
    use3d = cms.bool(False)
)


process.softPFElectronByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('any'),
    use3d = cms.bool(True)
)


process.softPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('any')
)


process.softPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string('any'),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz')
)


process.softPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('any'),
    use3d = cms.bool(False)
)


process.softPFMuonByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('any'),
    use3d = cms.bool(True)
)


process.softPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('any')
)


process.softPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string('any'),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz')
)


process.stripCPEESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentName = cms.string('stripCPE'),
    ComponentType = cms.string('SimpleStripCPE'),
    parameters = cms.PSet(

    )
)


process.trackCounting3D2ndComputer = cms.ESProducer("TrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(2),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.trackCounting3D3rdComputer = cms.ESProducer("TrackCountingESProducer",
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    deltaR = cms.double(-1.0),
    impactParameterType = cms.int32(0),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    maximumDecayLength = cms.double(5.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5),
    minimumImpactParameter = cms.double(-1),
    nthTrack = cms.int32(3),
    trackQualityClass = cms.string('any'),
    useSignedImpactParameterSig = cms.bool(True),
    useVariableJTA = cms.bool(False)
)


process.trackerGeometryDB = cms.ESProducer("TrackerDigiGeometryESModule",
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(True),
    fromDDD = cms.bool(False)
)


process.trackerNumberingGeometryDB = cms.ESProducer("TrackerGeometricDetESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(False)
)


process.trackerTopology = cms.ESProducer("TrackerTopologyEP",
    appendToDataLabel = cms.string('')
)


process.BTagRecord = cms.ESSource("EmptyESSource",
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('JetTagComputerRecord')
)


process.GlobalTag = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        enableConnectionSharing = cms.untracked.bool(True),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0)
    ),
    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
    globaltag = cms.string('75X_mcRun2_asymptotic_v1'),
    toGet = cms.VPSet()
)


process.eegeom = cms.ESSource("EmptyESSource",
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('EcalMappingRcd')
)


process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    GainWidthsForTrigPrims = cms.bool(False),
    HERecalibration = cms.bool(False),
    HEreCalibCutoff = cms.double(20.0),
    HFRecalibration = cms.bool(False),
    HcalReLabel = cms.PSet(
        RelabelHits = cms.untracked.bool(False),
        RelabelRules = cms.untracked.PSet(
            CorrectPhi = cms.untracked.bool(False),
            Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
                2, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
                3, 3, 4, 4, 4, 
                4, 4, 5, 5, 5, 
                5, 5, 5, 5)
        )
    ),
    hcalTopologyConstants = cms.PSet(
        maxDepthHB = cms.int32(2),
        maxDepthHE = cms.int32(3),
        mode = cms.string('HcalTopologyMode::LHC')
    ),
    iLumi = cms.double(-1.0),
    toGet = cms.untracked.vstring('GainWidths')
)


process.prefer("es_hardcode")

process.AnomalousCellParameters = cms.PSet(
    maxBadEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999)
)

process.CATopJetParameters = cms.PSet(
    algorithm = cms.int32(1),
    centralEtaCut = cms.double(2.5),
    debugLevel = cms.untracked.int32(0),
    deltarBins = cms.vdouble(0.19, 0.19, 0.19),
    etFrac = cms.double(0.7),
    jetCollInstanceName = cms.string('caTopSubJets'),
    nCellBins = cms.vdouble(1.9, 1.9, 1.9),
    ptFracBins = cms.vdouble(0.05, 0.05, 0.05),
    rBins = cms.vdouble(0.8, 0.8, 0.8),
    sumEtBins = cms.vdouble(0, 1600, 2600),
    sumEtEtaCut = cms.double(3.0),
    tagAlgo = cms.int32(0),
    useAdjacency = cms.int32(2),
    useMaxTower = cms.bool(False),
    verbose = cms.bool(False)
)

process.CondDBSetup = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        enableConnectionSharing = cms.untracked.bool(True),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0)
    )
)

process.EmptyJetIdParams = cms.PSet(
    Pt010_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt010_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt010_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt1020_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt1020_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt1020_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt2030_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt2030_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt2030_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt3050_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt3050_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
    Pt3050_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0)
)

process.GenJetParameters = cms.PSet(
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    Rho_EtaMax = cms.double(4.5),
    doAreaFastjet = cms.bool(False),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetPtMin = cms.double(3.0),
    jetType = cms.string('GenJet'),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("genParticlesForJets"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True)
)

process.HcalReLabel = cms.PSet(
    RelabelHits = cms.untracked.bool(False),
    RelabelRules = cms.untracked.PSet(
        CorrectPhi = cms.untracked.bool(False),
        Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3),
        Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
            2, 2, 2, 2, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3),
        Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
            3, 3, 4, 4, 4, 
            4, 4, 5, 5, 5, 
            5, 5, 5, 5)
    )
)

process.JetIdParams = cms.PSet(
    Pt010_Loose = cms.vdouble(0.0, 0.0, 0.0, 0.2),
    Pt010_Medium = cms.vdouble(0.2, 0.4, 0.2, 0.6),
    Pt010_Tight = cms.vdouble(0.5, 0.6, 0.6, 0.9),
    Pt1020_Loose = cms.vdouble(-0.4, -0.4, -0.4, 0.4),
    Pt1020_Medium = cms.vdouble(-0.3, 0.0, 0.0, 0.5),
    Pt1020_Tight = cms.vdouble(-0.2, 0.2, 0.2, 0.6),
    Pt2030_Loose = cms.vdouble(0.0, 0.0, 0.2, 0.6),
    Pt2030_Medium = cms.vdouble(0.2, 0.2, 0.5, 0.7),
    Pt2030_Tight = cms.vdouble(0.3, 0.4, 0.7, 0.8),
    Pt3050_Loose = cms.vdouble(0.0, 0.0, 0.6, 0.2),
    Pt3050_Medium = cms.vdouble(0.3, 0.2, 0.7, 0.8),
    Pt3050_Tight = cms.vdouble(0.5, 0.4, 0.8, 0.9)
)

process.METSignificanceParams = cms.PSet(
    dRMatch = cms.double(0.4),
    jetThreshold = cms.double(20),
    jeta = cms.vdouble(0.5, 1.1, 1.7, 2.3),
    jpar = cms.vdouble(1.41, 1.29, 1.41, 1.4, 2.53),
    phiResFile = cms.string('Spring10_PhiResolution_AK5PF.txt'),
    pjpar = cms.vdouble(0.0, 0.674),
    ptResFile = cms.string('Spring10_PtResolution_AK5PF.txt')
)

process.PFJetParameters = cms.PSet(
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    Rho_EtaMax = cms.double(4.4),
    doAreaDiskApprox = cms.bool(False),
    doAreaFastjet = cms.bool(False),
    doOutputJets = cms.bool(True),
    doPUOffsetCorr = cms.bool(False),
    doPVCorrection = cms.bool(False),
    doRhoFastjet = cms.bool(False),
    inputEMin = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    jetType = cms.string('PFJet'),
    minSeed = cms.uint32(14327),
    nSigmaPU = cms.double(1.0),
    radiusPU = cms.double(0.5),
    src = cms.InputTag("particleFlow"),
    srcPVs = cms.InputTag(""),
    useDeterministicSeed = cms.bool(True),
    voronoiRfact = cms.double(-0.9)
)

process.PhilV0 = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt010_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt010_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt1020_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt1020_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt1020_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt2030_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt2030_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt2030_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt3050_Loose = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt3050_Medium = cms.vdouble(-999.0, -999.0, -999.0, -999.0),
        Pt3050_Tight = cms.vdouble(-999.0, -999.0, -999.0, -999.0)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    tmvaMethod = cms.string('JetID'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/mva_JetID.weights.xml.gz'),
    version = cms.int32(0)
)

process.PhilV1 = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(0.0, 0.0, 0.0, 0.2),
        Pt010_Medium = cms.vdouble(0.2, 0.4, 0.2, 0.6),
        Pt010_Tight = cms.vdouble(0.5, 0.6, 0.6, 0.9),
        Pt1020_Loose = cms.vdouble(-0.4, -0.4, -0.4, 0.4),
        Pt1020_Medium = cms.vdouble(-0.3, 0.0, 0.0, 0.5),
        Pt1020_Tight = cms.vdouble(-0.2, 0.2, 0.2, 0.6),
        Pt2030_Loose = cms.vdouble(0.0, 0.0, 0.2, 0.6),
        Pt2030_Medium = cms.vdouble(0.2, 0.2, 0.5, 0.7),
        Pt2030_Tight = cms.vdouble(0.3, 0.4, 0.7, 0.8),
        Pt3050_Loose = cms.vdouble(0.0, 0.0, 0.6, 0.2),
        Pt3050_Medium = cms.vdouble(0.3, 0.2, 0.7, 0.8),
        Pt3050_Tight = cms.vdouble(0.5, 0.4, 0.8, 0.9)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('philv1'),
    tmvaMethod = cms.string('JetID'),
    tmvaSpectators = cms.vstring(),
    tmvaVariables = cms.vstring('nvtx', 
        'jetPt', 
        'jetEta', 
        'jetPhi', 
        'dZ', 
        'd0', 
        'beta', 
        'betaStar', 
        'nCharged', 
        'nNeutrals', 
        'dRMean', 
        'frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/mva_JetID_v1.weights.xml.gz'),
    version = cms.int32(-1)
)

process.PuJetIdCutBased_wp = cms.PSet(
    Pt010_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt010_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt010_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
    Pt010_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
    Pt010_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
    Pt010_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
    Pt1020_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt1020_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt1020_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
    Pt1020_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
    Pt1020_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
    Pt1020_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
    Pt2030_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt2030_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt2030_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
    Pt2030_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
    Pt2030_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
    Pt2030_RMSTight = cms.vdouble(0.05, 0.07, 0.03, 0.045),
    Pt3050_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt3050_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
    Pt3050_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
    Pt3050_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
    Pt3050_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
    Pt3050_RMSTight = cms.vdouble(0.05, 0.06, 0.03, 0.04)
)

process.PuJetIdMinMVA_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.9, -0.9, -0.94, -0.9),
    Pt010_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
    Pt010_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
    Pt1020_Loose = cms.vdouble(-0.9, -0.9, -0.94, -0.9),
    Pt1020_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
    Pt1020_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
    Pt2030_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
    Pt2030_Medium = cms.vdouble(0.1, -0.4, -0.5, -0.45),
    Pt2030_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0),
    Pt3050_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
    Pt3050_Medium = cms.vdouble(0.1, -0.4, -0.5, -0.45),
    Pt3050_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0)
)

process.PuJetIdOptMVA_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.9, -0.9, -0.9, -0.9),
    Pt010_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
    Pt010_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
    Pt1020_Loose = cms.vdouble(-0.9, -0.9, -0.9, -0.9),
    Pt1020_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
    Pt1020_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
    Pt2030_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
    Pt2030_Medium = cms.vdouble(0.1, -0.4, -0.4, -0.45),
    Pt2030_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0),
    Pt3050_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
    Pt3050_Medium = cms.vdouble(0.1, -0.4, -0.4, -0.45),
    Pt3050_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0)
)

process.combinedSecondaryVertexCommon = cms.PSet(
    charmCut = cms.double(1.5),
    correctVertexMass = cms.bool(True),
    minimumTrackWeight = cms.double(0.5),
    pseudoMultiplicityMin = cms.uint32(2),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    trackFlip = cms.bool(False),
    trackMultiplicityMin = cms.uint32(2),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig'),
    useTrackWeights = cms.bool(True),
    vertexFlip = cms.bool(False)
)

process.cutbased = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt010_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt010_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
        Pt010_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
        Pt010_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
        Pt010_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
        Pt1020_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt1020_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt1020_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
        Pt1020_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.07),
        Pt1020_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
        Pt1020_RMSTight = cms.vdouble(0.06, 0.07, 0.04, 0.05),
        Pt2030_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt2030_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt2030_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
        Pt2030_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
        Pt2030_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
        Pt2030_RMSTight = cms.vdouble(0.05, 0.07, 0.03, 0.045),
        Pt3050_BetaStarLoose = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt3050_BetaStarMedium = cms.vdouble(0.2, 0.3, 999.0, 999.0),
        Pt3050_BetaStarTight = cms.vdouble(0.15, 0.15, 999.0, 999.0),
        Pt3050_RMSLoose = cms.vdouble(0.06, 0.05, 0.05, 0.055),
        Pt3050_RMSMedium = cms.vdouble(0.06, 0.03, 0.03, 0.04),
        Pt3050_RMSTight = cms.vdouble(0.05, 0.06, 0.03, 0.04)
    ),
    cutBased = cms.bool(True),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('cutbased')
)

process.full = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.9, -0.9, -0.9, -0.9),
        Pt010_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
        Pt010_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
        Pt1020_Loose = cms.vdouble(-0.9, -0.9, -0.9, -0.9),
        Pt1020_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
        Pt1020_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
        Pt2030_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
        Pt2030_Medium = cms.vdouble(0.1, -0.4, -0.4, -0.45),
        Pt2030_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0),
        Pt3050_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
        Pt3050_Medium = cms.vdouble(0.1, -0.4, -0.4, -0.45),
        Pt3050_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('full'),
    tmvaMethod = cms.string('PuJetIdOptMVA'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta'),
    tmvaVariables = cms.vstring('frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05', 
        'nvtx', 
        'nNeutrals', 
        'beta', 
        'betaStar', 
        'dZ', 
        'nCharged'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassification_PuJetIdOptMVA.weights.xml.gz'),
    version = cms.int32(-1)
)

process.full_53x = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
        Pt010_MET = cms.vdouble(0.0, -0.6, -0.4, -0.4),
        Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
        Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
        Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
        Pt1020_MET = cms.vdouble(0.3, -0.2, -0.4, -0.4),
        Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
        Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
        Pt2030_Loose = cms.vdouble(-0.63, -0.6, -0.55, -0.45),
        Pt2030_MET = cms.vdouble(0.0, 0.0, 0.0, 0.0),
        Pt2030_Medium = cms.vdouble(0.1, -0.36, -0.54, -0.54),
        Pt2030_Tight = cms.vdouble(0.73, 0.05, -0.26, -0.42),
        Pt3050_Loose = cms.vdouble(-0.63, -0.6, -0.55, -0.45),
        Pt3050_MET = cms.vdouble(0.0, 0.0, -0.1, -0.2),
        Pt3050_Medium = cms.vdouble(0.1, -0.36, -0.54, -0.54),
        Pt3050_Tight = cms.vdouble(0.73, 0.05, -0.26, -0.42)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('full53x'),
    tmvaMethod = cms.string('JetIDMVAHighPt'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta', 
        'jetPhi'),
    tmvaVariables = cms.vstring('nvtx', 
        'dZ', 
        'beta', 
        'betaStar', 
        'nCharged', 
        'nNeutrals', 
        'dR2Mean', 
        'ptD', 
        'frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05'),
    tmvaWeights = cms.string('CondFormats/JetMETObjects/data/TMVAClassificationCategory_JetID_53X_Dec2012.weights.xml'),
    version = cms.int32(-1)
)

process.full_53x_2015 = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
        Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
        Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
        Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
        Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
        Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
        Pt2030_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
        Pt2030_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
        Pt2030_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17),
        Pt3050_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
        Pt3050_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
        Pt3050_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('full'),
    tmvaMethod = cms.string('JetIDMVAHighPt'),
    tmvaSpectators = cms.vstring('p4.fCoordinates.fPt', 
        'p4.fCoordinates.fEta', 
        'nTrueInt', 
        'dRMatch'),
    tmvaVariables = cms.vstring('DRweighted', 
        'rho', 
        'nTot', 
        'nCh', 
        'axisMajor', 
        'axisMinor', 
        'fRing0', 
        'fRing1', 
        'fRing2', 
        'fRing3', 
        'ptD', 
        'beta', 
        'betaStar', 
        'min(pull,0.1)', 
        'jetR', 
        'jetRchg'),
    tmvaWeights_jteta_0_2 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_0_2.xml'),
    tmvaWeights_jteta_2_2p5 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_2_2p5.xml'),
    tmvaWeights_jteta_2p5_3 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_2p5_3.xml'),
    tmvaWeights_jteta_3_5 = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_BDTG.weights_jteta_3_5.xml'),
    version = cms.int32(-1)
)

process.full_53x_chs = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
        Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
        Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
        Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
        Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
        Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
        Pt2030_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
        Pt2030_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
        Pt2030_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17),
        Pt3050_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
        Pt3050_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
        Pt3050_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('full'),
    tmvaMethod = cms.string('JetIDMVAHighPt'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta', 
        'jetPhi'),
    tmvaVariables = cms.vstring('nvtx', 
        'dZ', 
        'beta', 
        'betaStar', 
        'nCharged', 
        'nNeutrals', 
        'dR2Mean', 
        'ptD', 
        'frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05'),
    tmvaWeights = cms.string('CondFormats/JetMETObjects/data/TMVAClassificationCategory_JetID_53X_chs_Dec2012.weights.xml'),
    version = cms.int32(-1)
)

process.full_53x_chs_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
    Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
    Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
    Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
    Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
    Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
    Pt2030_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
    Pt2030_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
    Pt2030_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17),
    Pt3050_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
    Pt3050_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
    Pt3050_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17)
)

process.full_53x_chs_wp_2015 = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
    Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
    Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
    Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
    Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
    Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
    Pt2030_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
    Pt2030_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
    Pt2030_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17),
    Pt3050_Loose = cms.vdouble(-0.15, -0.26, -0.16, -0.16),
    Pt3050_Medium = cms.vdouble(-0.07, -0.09, 0.0, -0.06),
    Pt3050_Tight = cms.vdouble(0.78, 0.5, 0.17, 0.17)
)

process.full_53x_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
    Pt010_MET = cms.vdouble(0.0, -0.6, -0.4, -0.4),
    Pt010_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
    Pt010_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
    Pt1020_Loose = cms.vdouble(-0.95, -0.96, -0.94, -0.95),
    Pt1020_MET = cms.vdouble(0.3, -0.2, -0.4, -0.4),
    Pt1020_Medium = cms.vdouble(-0.83, -0.92, -0.9, -0.92),
    Pt1020_Tight = cms.vdouble(-0.83, -0.81, -0.74, -0.81),
    Pt2030_Loose = cms.vdouble(-0.63, -0.6, -0.55, -0.45),
    Pt2030_MET = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    Pt2030_Medium = cms.vdouble(0.1, -0.36, -0.54, -0.54),
    Pt2030_Tight = cms.vdouble(0.73, 0.05, -0.26, -0.42),
    Pt3050_Loose = cms.vdouble(-0.63, -0.6, -0.55, -0.45),
    Pt3050_MET = cms.vdouble(0.0, 0.0, -0.1, -0.2),
    Pt3050_Medium = cms.vdouble(0.1, -0.36, -0.54, -0.54),
    Pt3050_Tight = cms.vdouble(0.73, 0.05, -0.26, -0.42)
)

process.full_5x = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.95, -0.97, -0.97, -0.97),
        Pt010_Medium = cms.vdouble(-0.83, -0.96, -0.95, -0.96),
        Pt010_Tight = cms.vdouble(-0.47, -0.92, -0.92, -0.94),
        Pt1020_Loose = cms.vdouble(-0.95, -0.97, -0.97, -0.97),
        Pt1020_Medium = cms.vdouble(-0.83, -0.96, -0.95, -0.96),
        Pt1020_Tight = cms.vdouble(-0.47, -0.92, -0.92, -0.94),
        Pt2030_Loose = cms.vdouble(-0.8, -0.85, -0.84, -0.85),
        Pt2030_Medium = cms.vdouble(-0.4, -0.74, -0.76, -0.81),
        Pt2030_Tight = cms.vdouble(0.32, -0.49, -0.61, -0.74),
        Pt3050_Loose = cms.vdouble(-0.8, -0.85, -0.84, -0.85),
        Pt3050_Medium = cms.vdouble(-0.4, -0.74, -0.76, -0.81),
        Pt3050_Tight = cms.vdouble(0.32, -0.49, -0.61, -0.74)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('full'),
    tmvaMethod = cms.string('BDT_fullPlusRMS'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta'),
    tmvaVariables = cms.vstring('frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05', 
        'dR2Mean', 
        'nvtx', 
        'nNeutrals', 
        'beta', 
        'betaStar', 
        'dZ', 
        'nCharged'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassification_5x_BDT_fullPlusRMS.weights.xml.gz'),
    version = cms.int32(-1)
)

process.full_5x_chs = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.98, -0.95, -0.94, -0.94),
        Pt010_Medium = cms.vdouble(-0.94, -0.91, -0.91, -0.92),
        Pt010_Tight = cms.vdouble(-0.59, -0.75, -0.78, -0.8),
        Pt1020_Loose = cms.vdouble(-0.98, -0.95, -0.94, -0.94),
        Pt1020_Medium = cms.vdouble(-0.94, -0.91, -0.91, -0.92),
        Pt1020_Tight = cms.vdouble(-0.59, -0.75, -0.78, -0.8),
        Pt2030_Loose = cms.vdouble(-0.89, -0.77, -0.69, -0.75),
        Pt2030_Medium = cms.vdouble(-0.58, -0.65, -0.57, -0.67),
        Pt2030_Tight = cms.vdouble(0.41, -0.1, -0.2, -0.45),
        Pt3050_Loose = cms.vdouble(-0.89, -0.77, -0.69, -0.57),
        Pt3050_Medium = cms.vdouble(-0.58, -0.65, -0.57, -0.67),
        Pt3050_Tight = cms.vdouble(0.41, -0.1, -0.2, -0.45)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('full'),
    tmvaMethod = cms.string('BDT_chsFullPlusRMS'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta'),
    tmvaVariables = cms.vstring('frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05', 
        'dR2Mean', 
        'nvtx', 
        'nNeutrals', 
        'beta', 
        'betaStar', 
        'dZ', 
        'nCharged'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassification_5x_BDT_chsFullPlusRMS.weights.xml.gz'),
    version = cms.int32(-1)
)

process.full_5x_chs_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.98, -0.95, -0.94, -0.94),
    Pt010_Medium = cms.vdouble(-0.94, -0.91, -0.91, -0.92),
    Pt010_Tight = cms.vdouble(-0.59, -0.75, -0.78, -0.8),
    Pt1020_Loose = cms.vdouble(-0.98, -0.95, -0.94, -0.94),
    Pt1020_Medium = cms.vdouble(-0.94, -0.91, -0.91, -0.92),
    Pt1020_Tight = cms.vdouble(-0.59, -0.75, -0.78, -0.8),
    Pt2030_Loose = cms.vdouble(-0.89, -0.77, -0.69, -0.75),
    Pt2030_Medium = cms.vdouble(-0.58, -0.65, -0.57, -0.67),
    Pt2030_Tight = cms.vdouble(0.41, -0.1, -0.2, -0.45),
    Pt3050_Loose = cms.vdouble(-0.89, -0.77, -0.69, -0.57),
    Pt3050_Medium = cms.vdouble(-0.58, -0.65, -0.57, -0.67),
    Pt3050_Tight = cms.vdouble(0.41, -0.1, -0.2, -0.45)
)

process.full_5x_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.95, -0.97, -0.97, -0.97),
    Pt010_Medium = cms.vdouble(-0.83, -0.96, -0.95, -0.96),
    Pt010_Tight = cms.vdouble(-0.47, -0.92, -0.92, -0.94),
    Pt1020_Loose = cms.vdouble(-0.95, -0.97, -0.97, -0.97),
    Pt1020_Medium = cms.vdouble(-0.83, -0.96, -0.95, -0.96),
    Pt1020_Tight = cms.vdouble(-0.47, -0.92, -0.92, -0.94),
    Pt2030_Loose = cms.vdouble(-0.8, -0.85, -0.84, -0.85),
    Pt2030_Medium = cms.vdouble(-0.4, -0.74, -0.76, -0.81),
    Pt2030_Tight = cms.vdouble(0.32, -0.49, -0.61, -0.74),
    Pt3050_Loose = cms.vdouble(-0.8, -0.85, -0.84, -0.85),
    Pt3050_Medium = cms.vdouble(-0.4, -0.74, -0.76, -0.81),
    Pt3050_Tight = cms.vdouble(0.32, -0.49, -0.61, -0.74)
)

process.ghostTrackCommon = cms.PSet(
    charmCut = cms.double(1.5),
    minimumTrackWeight = cms.double(0.5),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    ),
    trackSort = cms.string('sip2dSig')
)

process.ghostTrackVertexRecoBlock = cms.PSet(
    vertexReco = cms.PSet(
        finder = cms.string('gtvr'),
        fitType = cms.string('RefitGhostTrackWithVertices'),
        maxFitChi2 = cms.double(10.0),
        mergeThreshold = cms.double(3.0),
        primcut = cms.double(2.0),
        seccut = cms.double(4.0)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.met_53x = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-2, -2, -2, -2, -2),
        Pt010_MET = cms.vdouble(-0.2, -0.3, -0.5, -0.5),
        Pt010_Medium = cms.vdouble(-2, -2, -2, -2, -2),
        Pt010_Tight = cms.vdouble(-2, -2, -2, -2, -2),
        Pt1020_Loose = cms.vdouble(-2, -2, -2, -2, -2),
        Pt1020_MET = cms.vdouble(-0.2, -0.2, -0.5, -0.3),
        Pt1020_Medium = cms.vdouble(-2, -2, -2, -2, -2),
        Pt1020_Tight = cms.vdouble(-2, -2, -2, -2, -2),
        Pt2030_Loose = cms.vdouble(-2, -2, -2, -2, -2),
        Pt2030_MET = cms.vdouble(-0.2, -0.2, -0.2, 0.1),
        Pt2030_Medium = cms.vdouble(-2, -2, -2, -2, -2),
        Pt2030_Tight = cms.vdouble(-2, -2, -2, -2, -2),
        Pt3050_Loose = cms.vdouble(-2, -2, -2, -2, -2),
        Pt3050_MET = cms.vdouble(-0.2, -0.2, 0.0, 0.2),
        Pt3050_Medium = cms.vdouble(-2, -2, -2, -2, -2),
        Pt3050_Tight = cms.vdouble(-2, -2, -2, -2, -2)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('met53x'),
    tmvaMethod = cms.string('JetIDMVAMET'),
    tmvaSpectators = cms.vstring(),
    tmvaVariables = cms.vstring('nvtx', 
        'jetPt', 
        'jetEta', 
        'jetPhi', 
        'dZ', 
        'beta', 
        'betaStar', 
        'nCharged', 
        'nNeutrals', 
        'dR2Mean', 
        'ptD', 
        'frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_MET_53X_Dec2012.weights.xml.gz'),
    version = cms.int32(-1)
)

process.met_53x_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-2, -2, -2, -2, -2),
    Pt010_MET = cms.vdouble(-0.2, -0.3, -0.5, -0.5),
    Pt010_Medium = cms.vdouble(-2, -2, -2, -2, -2),
    Pt010_Tight = cms.vdouble(-2, -2, -2, -2, -2),
    Pt1020_Loose = cms.vdouble(-2, -2, -2, -2, -2),
    Pt1020_MET = cms.vdouble(-0.2, -0.2, -0.5, -0.3),
    Pt1020_Medium = cms.vdouble(-2, -2, -2, -2, -2),
    Pt1020_Tight = cms.vdouble(-2, -2, -2, -2, -2),
    Pt2030_Loose = cms.vdouble(-2, -2, -2, -2, -2),
    Pt2030_MET = cms.vdouble(-0.2, -0.2, -0.2, 0.1),
    Pt2030_Medium = cms.vdouble(-2, -2, -2, -2, -2),
    Pt2030_Tight = cms.vdouble(-2, -2, -2, -2, -2),
    Pt3050_Loose = cms.vdouble(-2, -2, -2, -2, -2),
    Pt3050_MET = cms.vdouble(-0.2, -0.2, 0.0, 0.2),
    Pt3050_Medium = cms.vdouble(-2, -2, -2, -2, -2),
    Pt3050_Tight = cms.vdouble(-2, -2, -2, -2, -2)
)

process.metfull_53x_wp = cms.PSet(
    Pt010_MET = cms.vdouble(-0.2, -0.3, -0.5, -0.5),
    Pt1020_MET = cms.vdouble(-0.2, -0.2, -0.5, -0.3),
    Pt2030_MET = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    Pt3050_MET = cms.vdouble(0.0, 0.0, -0.1, -0.2)
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.simple = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.9, -0.9, -0.94, -0.9),
        Pt010_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
        Pt010_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
        Pt1020_Loose = cms.vdouble(-0.9, -0.9, -0.94, -0.9),
        Pt1020_Medium = cms.vdouble(-0.73, -0.89, -0.89, -0.83),
        Pt1020_Tight = cms.vdouble(-0.5, -0.2, -0.83, -0.7),
        Pt2030_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
        Pt2030_Medium = cms.vdouble(0.1, -0.4, -0.5, -0.45),
        Pt2030_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0),
        Pt3050_Loose = cms.vdouble(-0.4, -0.85, -0.7, -0.6),
        Pt3050_Medium = cms.vdouble(0.1, -0.4, -0.5, -0.45),
        Pt3050_Tight = cms.vdouble(-0.2, 0.0, 0.0, 0.0)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('simple'),
    tmvaMethod = cms.string('PuJetIdMinMVA'),
    tmvaSpectators = cms.vstring('nvtx', 
        'jetPt', 
        'jetEta'),
    tmvaVariables = cms.vstring('frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05', 
        'beta', 
        'betaStar'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassification_PuJetIdMinMVA.weights.xml.gz'),
    version = cms.int32(-1)
)

process.simple_5x = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.95, -0.97, -0.96, -0.97),
        Pt010_Medium = cms.vdouble(-0.85, -0.96, -0.95, -0.96),
        Pt010_Tight = cms.vdouble(-0.54, -0.93, -0.93, -0.94),
        Pt1020_Loose = cms.vdouble(-0.95, -0.97, -0.96, -0.97),
        Pt1020_Medium = cms.vdouble(-0.85, -0.96, -0.95, -0.96),
        Pt1020_Tight = cms.vdouble(-0.54, -0.93, -0.93, -0.94),
        Pt2030_Loose = cms.vdouble(-0.8, -0.86, -0.8, -0.84),
        Pt2030_Medium = cms.vdouble(-0.4, -0.73, -0.74, -0.8),
        Pt2030_Tight = cms.vdouble(0.26, -0.54, -0.63, -0.74),
        Pt3050_Loose = cms.vdouble(-0.8, -0.86, -0.8, -0.84),
        Pt3050_Medium = cms.vdouble(-0.4, -0.73, -0.74, -0.8),
        Pt3050_Tight = cms.vdouble(0.26, -0.54, -0.63, -0.74)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('simple'),
    tmvaMethod = cms.string('BDT_simpleNoVtxCat'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta'),
    tmvaVariables = cms.vstring('frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05', 
        'nvtx', 
        'beta', 
        'betaStar'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassification_5x_BDT_simpleNoVtxCat.weights.xml.gz'),
    version = cms.int32(-1)
)

process.simple_5x_chs = cms.PSet(
    JetIdParams = cms.PSet(
        Pt010_Loose = cms.vdouble(-0.98, -0.96, -0.94, -0.94),
        Pt010_Medium = cms.vdouble(-0.95, -0.94, -0.92, -0.91),
        Pt010_Tight = cms.vdouble(-0.6, -0.74, -0.78, -0.81),
        Pt1020_Loose = cms.vdouble(-0.98, -0.96, -0.94, -0.94),
        Pt1020_Medium = cms.vdouble(-0.95, -0.94, -0.92, -0.91),
        Pt1020_Tight = cms.vdouble(-0.6, -0.74, -0.78, -0.81),
        Pt2030_Loose = cms.vdouble(-0.89, -0.75, -0.72, -0.75),
        Pt2030_Medium = cms.vdouble(-0.59, -0.65, -0.56, -0.68),
        Pt2030_Tight = cms.vdouble(-0.47, -0.06, -0.23, -0.47),
        Pt3050_Loose = cms.vdouble(-0.89, -0.75, -0.72, -0.75),
        Pt3050_Medium = cms.vdouble(-0.59, -0.65, -0.56, -0.68),
        Pt3050_Tight = cms.vdouble(-0.47, -0.06, -0.23, -0.47)
    ),
    cutBased = cms.bool(False),
    impactParTkThreshold = cms.double(1.0),
    label = cms.string('simple'),
    tmvaMethod = cms.string('BDT_chsSimpleNoVtxCat'),
    tmvaSpectators = cms.vstring('jetPt', 
        'jetEta'),
    tmvaVariables = cms.vstring('frac01', 
        'frac02', 
        'frac03', 
        'frac04', 
        'frac05', 
        'nvtx', 
        'beta', 
        'betaStar'),
    tmvaWeights = cms.string('RecoJets/JetProducers/data/TMVAClassification_5x_BDT_chsSimpleNoVtxCat.weights.xml.gz'),
    version = cms.int32(-1)
)

process.simple_5x_chs_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.98, -0.96, -0.94, -0.94),
    Pt010_Medium = cms.vdouble(-0.95, -0.94, -0.92, -0.91),
    Pt010_Tight = cms.vdouble(-0.6, -0.74, -0.78, -0.81),
    Pt1020_Loose = cms.vdouble(-0.98, -0.96, -0.94, -0.94),
    Pt1020_Medium = cms.vdouble(-0.95, -0.94, -0.92, -0.91),
    Pt1020_Tight = cms.vdouble(-0.6, -0.74, -0.78, -0.81),
    Pt2030_Loose = cms.vdouble(-0.89, -0.75, -0.72, -0.75),
    Pt2030_Medium = cms.vdouble(-0.59, -0.65, -0.56, -0.68),
    Pt2030_Tight = cms.vdouble(-0.47, -0.06, -0.23, -0.47),
    Pt3050_Loose = cms.vdouble(-0.89, -0.75, -0.72, -0.75),
    Pt3050_Medium = cms.vdouble(-0.59, -0.65, -0.56, -0.68),
    Pt3050_Tight = cms.vdouble(-0.47, -0.06, -0.23, -0.47)
)

process.simple_5x_wp = cms.PSet(
    Pt010_Loose = cms.vdouble(-0.95, -0.97, -0.96, -0.97),
    Pt010_Medium = cms.vdouble(-0.85, -0.96, -0.95, -0.96),
    Pt010_Tight = cms.vdouble(-0.54, -0.93, -0.93, -0.94),
    Pt1020_Loose = cms.vdouble(-0.95, -0.97, -0.96, -0.97),
    Pt1020_Medium = cms.vdouble(-0.85, -0.96, -0.95, -0.96),
    Pt1020_Tight = cms.vdouble(-0.54, -0.93, -0.93, -0.94),
    Pt2030_Loose = cms.vdouble(-0.8, -0.86, -0.8, -0.84),
    Pt2030_Medium = cms.vdouble(-0.4, -0.73, -0.74, -0.8),
    Pt2030_Tight = cms.vdouble(0.26, -0.54, -0.63, -0.74),
    Pt3050_Loose = cms.vdouble(-0.8, -0.86, -0.8, -0.84),
    Pt3050_Medium = cms.vdouble(-0.4, -0.73, -0.74, -0.8),
    Pt3050_Tight = cms.vdouble(0.26, -0.54, -0.63, -0.74)
)

process.trackPseudoSelectionBlock = cms.PSet(
    trackPseudoSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    )
)

process.trackSelectionBlock = cms.PSet(
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(5),
        maxDistToAxis = cms.double(0.07),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(0),
        ptMin = cms.double(0.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(0),
        useVariableJTA = cms.bool(False)
    )
)

process.variableJTAPars = cms.PSet(
    a_dR = cms.double(-0.001053),
    a_pT = cms.double(0.005263),
    b_dR = cms.double(0.6263),
    b_pT = cms.double(0.3684),
    max_pT = cms.double(500),
    max_pT_dRcut = cms.double(0.1),
    max_pT_trackPTcut = cms.double(3),
    min_pT = cms.double(120),
    min_pT_dRcut = cms.double(0.5)
)

process.vertexCutsBlock = cms.PSet(
    vertexCuts = cms.PSet(
        distSig2dMax = cms.double(99999.9),
        distSig2dMin = cms.double(3.0),
        distSig3dMax = cms.double(99999.9),
        distSig3dMin = cms.double(-99999.9),
        distVal2dMax = cms.double(2.5),
        distVal2dMin = cms.double(0.01),
        distVal3dMax = cms.double(99999.9),
        distVal3dMin = cms.double(-99999.9),
        fracPV = cms.double(0.65),
        massMax = cms.double(6.5),
        maxDeltaRToJetAxis = cms.double(0.4),
        minimumTrackWeight = cms.double(0.5),
        multiplicityMin = cms.uint32(2),
        useTrackWeights = cms.bool(True),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        )
    )
)

process.vertexRecoBlock = cms.PSet(
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        weightthreshold = cms.double(0.001)
    )
)

process.vertexSelectionBlock = cms.PSet(
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    )
)

process.vertexTrackSelectionBlock = cms.PSet(
    trackSelection = cms.PSet(
        a_dR = cms.double(-0.001053),
        a_pT = cms.double(0.005263),
        b_dR = cms.double(0.6263),
        b_pT = cms.double(0.3684),
        jetDeltaRMax = cms.double(0.3),
        maxDecayLen = cms.double(99999.9),
        maxDistToAxis = cms.double(0.2),
        max_pT = cms.double(500),
        max_pT_dRcut = cms.double(0.1),
        max_pT_trackPTcut = cms.double(3),
        min_pT = cms.double(120),
        min_pT_dRcut = cms.double(0.5),
        normChi2Max = cms.double(99999.9),
        pixelHitsMin = cms.uint32(2),
        ptMin = cms.double(1.0),
        qualityClass = cms.string('any'),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip2dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip3dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip3dValMin = cms.double(-99999.9),
        totalHitsMin = cms.uint32(8),
        useVariableJTA = cms.bool(False)
    )
)


