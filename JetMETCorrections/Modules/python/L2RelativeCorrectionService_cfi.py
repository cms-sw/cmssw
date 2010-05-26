# Example cfi file for the L2Relative correction service. 
# It is used for the HLT confguration database.
import FWCore.ParameterSet.Config as cms
L2JetCorrectorIC5Calo = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('Summer08Redigi_L2Relative_IC5Calo'),
    label = cms.string('L2RelativeJetCorrectorIC5Calo')
)
