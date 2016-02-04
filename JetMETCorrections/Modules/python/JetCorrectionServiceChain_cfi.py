# Example cfi file for the L2L3 chain correction service. 
# It is used for the HLT confguration database.
import FWCore.ParameterSet.Config as cms
L2L3JetCorrectorIC5Calo = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorIC5Calo','L3AbsoluteJetCorrectorIC5Calo'),
    label = cms.string('L2L3JetCorrectorIC5Calo') 
)
