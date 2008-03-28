import FWCore.ParameterSet.Config as cms

# File: JetCorrectionsCSA06mcone7.cff
# Author: R. Harris
# Date: 1/09/07
#
# Jet corrections for midpoint cone R=0.7 jets from CSA06.
# 
corJetMcone7 = cms.EDFilter("MCJet",
    src = cms.InputTag("midPointCone7CaloJets"),
    tagName = cms.string('CSA06_Midpoint_Cone_07')
)

JetCorrectionsMcone7 = cms.Sequence(corJetMcone7)

