import FWCore.ParameterSet.Config as cms

# Jet corrections.
#
# Define the correction services for each algorithm.
#
# Service
ZSPJetCorrectorIcone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.vstring('ZSP_CMSSW22X_Iterative_Cone_05_PU0','ZSP_CMSSW22X_Iterative_Cone_05_PU1','ZSP_CMSSW22X_Iterative_Cone_05_PU2','ZSP_CMSSW22X_Iterative_Cone_05_PU5'),
    tagNameOffset = cms.vstring('L1Offset_Noise_IC5Calo','L1Offset_1PU_IC5Calo','L1Offset_2PU_IC5Calo','L1Offset_5PU_IC5Calo'),
    label = cms.string('ZSPJetCorrectorIcone5'),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0)
)
ZSPJetCorrectorSiscone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.vstring('ZSP_CMSSW22X_Iterative_Cone_05_PU0','ZSP_CMSSW22X_Iterative_Cone_05_PU1','ZSP_CMSSW22X_Iterative_Cone_05_PU2','ZSP_CMSSW22X_Iterative_Cone_05_PU5'),
    tagNameOffset = cms.vstring('L1Offset_Noise_IC5Calo','L1Offset_1PU_IC5Calo','L1Offset_2PU_IC5Calo','L1Offset_5PU_IC5Calo'),
    label = cms.string('ZSPJetCorrectorSiscone5'),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0)
)
ZSPJetCorrectorAntiKt5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.vstring('ZSP_CMSSW22X_Iterative_Cone_05_PU0','ZSP_CMSSW22X_Iterative_Cone_05_PU1','ZSP_CMSSW22X_Iterative_Cone_05_PU2','ZSP_CMSSW22X_Iterative_Cone_05_PU5'),
    tagNameOffset = cms.vstring('L1Offset_Noise_IC5Calo','L1Offset_1PU_IC5Calo','L1Offset_2PU_IC5Calo','L1Offset_5PU_IC5Calo'),
    label = cms.string('ZSPJetCorrectorAntiKt5'),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0)
)

# Modules
#   
#   Define the producers of corrected jet collections for each algorithm.
#
ZSPJetCorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('ZSPJetCorrectorIcone5'),
    alias = cms.untracked.string('ZSPJetCorJetIcone5')
)
ZSPJetCorJetSiscone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    correctors = cms.vstring('ZSPJetCorrectorSiscone5'),
    alias = cms.untracked.string('ZSPJetCorJetSiscone5')
)
ZSPJetCorJetAntiKt5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("ak5CaloJets"),
    correctors = cms.vstring('ZSPJetCorrectorAntiKt5'),
    alias = cms.untracked.string('ZSPJetCorJetAntiKt5')
)

#
#  Define a sequence to make all corrected jet collections at once.
#
ZSPJetCorrectionsIcone5 = cms.Sequence(ZSPJetCorJetIcone5)
ZSPJetCorrectionsSisCone5 = cms.Sequence(ZSPJetCorJetSiscone5)
ZSPJetCorrectionsAntiKt5 = cms.Sequence(ZSPJetCorJetAntiKt5)

# For backward-compatiblity (but to be deprecated!)
ZSPJetCorrections = ZSPJetCorrectionsIcone5
