import FWCore.ParameterSet.Config as cms

# Jet corrections.
#
# Define the correction services for each algorithm.
#
ZSPJetCorrectorIcone5 = cms.ESSource("ZSPJetCorrectionService",
    tagName = cms.string('ZSP_CMSSW152_Iterative_Cone_05'),
    label = cms.string('ZSPJetCorrectorIcone5')
)

#   
#   Define the producers of corrected jet collections for each algorithm.
#
ZSPJetCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('ZSPJetCorrectorIcone5'),
    alias = cms.untracked.string('ZSPJetCorJetIcone5')
)

#
#  Define a sequence to make all corrected jet collections at once.
#
ZSPJetCorrections = cms.Sequence(ZSPJetCorJetIcone5)

