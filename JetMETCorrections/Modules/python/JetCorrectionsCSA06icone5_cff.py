import FWCore.ParameterSet.Config as cms

# File: JetCorrectionsCSA06icone5.cff
# Author: R. Harris
# Date: 12/28/06
#
# Jet corrections for iterative cone R=0.5 jets from CSA06.
# 
corJetIcone5 = cms.EDFilter("MCJet",
    src = cms.InputTag("iterativeCone5CaloJets"),
    tagName = cms.string('CSA06_Iterative_Cone_05')
)

JetCorrectionsIcone5 = cms.Sequence(corJetIcone5)

