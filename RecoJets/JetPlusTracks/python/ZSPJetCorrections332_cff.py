import FWCore.ParameterSet.Config as cms
# Modules
#   
#   Define the producers of corrected jet collections for each algorithm.
#
ZSPJetCorJetIcone5 = cms.EDProducer("CaloJetProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    alias = cms.untracked.string('ZSPJetCorJetIcone5')
)
ZSPJetCorJetSiscone5 = cms.EDProducer("CaloJetProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
    alias = cms.untracked.string('ZSPJetCorJetSiscone5')
)
ZSPJetCorJetAntiKt5 = cms.EDProducer("CaloJetProducer",
    src = cms.InputTag("ak5CaloJets"),
    tagName = cms.vstring('ZSP_CMSSW332_Iterative_Cone_05_PU0'),
    tagNameOffset = cms.vstring(),
    PU = cms.int32(-1),
    FixedPU = cms.int32(0),
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
