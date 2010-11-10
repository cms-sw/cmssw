# Example cfi file for the L3Absolute correction service. 
# It is used for the HLT confguration database.
import FWCore.ParameterSet.Config as cms
L3JetCorrectorIC5Calo = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('Summer08Redigi_L3Absolute_IC5Calo'),
    label = cms.string('L3AbsoluteJetCorrectorIC5Calo')
)
