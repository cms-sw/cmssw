import FWCore.ParameterSet.Config as cms

# Jet corrections.
#
# Define the correction services for each algorithm.
#
# Service
ZSPJetCorrectorIcone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.string('ZSP_CMSSW219_Iterative_Cone_05'),
    label = cms.string('ZSPJetCorrectorIcone5')
#    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
)
ZSPJetCorrectorSiscone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.string('ZSP_CMSSW219_Iterative_Cone_05'),
    label = cms.string('ZSPJetCorrectorSiscone5')
#    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
)
ZSPJetCorrectorAntiKt5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.string('ZSP_CMSSW219_Iterative_Cone_05'),
    label = cms.string('ZSPJetCorrectorAntiKt5')
#    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
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
