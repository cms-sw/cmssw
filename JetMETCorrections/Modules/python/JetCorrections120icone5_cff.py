import FWCore.ParameterSet.Config as cms

# File: JetCorrections120icone5.cff
# Author: R. Harris
# Date: 1/30/07
#
# Jet corrections for iterative cone R=0.5 jets from CMSSW_1_2_0.
# 
corJetIcone5 = cms.EDFilter("MCJet",
    src = cms.InputTag("iterativeCone5CaloJets"),
    tagName = cms.string('CMSSW_120_Iterative_Cone_05')
)

JetCorrectionsIcone5 = cms.Sequence(corJetIcone5)

