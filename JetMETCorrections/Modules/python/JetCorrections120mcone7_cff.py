import FWCore.ParameterSet.Config as cms

# File: JetCorrections120mcone7.cff
# Author: R. Harris
# Date: 1/30/07
#
# Jet corrections for midpoint cone R=0.7 jets from CMSSW_1_2_0.
# 
corJetMcone7 = cms.EDFilter("MCJet",
    src = cms.InputTag("midPointCone7CaloJets"),
    tagName = cms.string('CMSSW_120_Midpoint_Cone_07')
)

JetCorrectionsMcone7 = cms.Sequence(corJetMcone7)

