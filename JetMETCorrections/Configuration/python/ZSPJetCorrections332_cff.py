import FWCore.ParameterSet.Config as cms

# Jet corrections.
#
# Define the correction services for each algorithm.
#
# Service
ZSPJetCorrectorIcone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    label = cms.string('ZSPJetCorrectorIcone5'),
#    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0)
)
ZSPJetCorrectorSiscone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    label = cms.string('ZSPJetCorrectorSiscone5'),
#    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0)
)
ZSPJetCorrectorAntiKt5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    label = cms.string('ZSPJetCorrectorAntiKt5'),
#    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
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
