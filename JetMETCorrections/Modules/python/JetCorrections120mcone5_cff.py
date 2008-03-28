import FWCore.ParameterSet.Config as cms

# File: JetCorrections120mcone5.cff
# Author: R. Harris
# Date: 1/30/07
#
# Jet corrections for midpoint cone R=0.5 jets from CMSSW_1_2_0
# 
corJetMcone5 = cms.EDFilter("MCJet",
    src = cms.InputTag("midPointCone5CaloJets"),
    tagName = cms.string('CMSSW_120_Midpoint_Cone_05')
)

JetCorrectionsMcone5 = cms.Sequence(corJetMcone5)

