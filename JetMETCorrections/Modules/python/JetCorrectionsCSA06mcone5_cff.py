import FWCore.ParameterSet.Config as cms

# File: JetCorrectionsCSA06mcone5.cff
# Author: R. Harris
# Date: 12/06/06
#
# Jet corrections for midpoint cone R=0.5 jets from CSA06.
# 
corJetMcone5 = cms.EDFilter("MCJet",
    src = cms.InputTag("midPointCone5CaloJets"),
    tagName = cms.string('CSA06_Midpoint_Cone_05')
)

JetCorrectionsMcone5 = cms.Sequence(corJetMcone5)

