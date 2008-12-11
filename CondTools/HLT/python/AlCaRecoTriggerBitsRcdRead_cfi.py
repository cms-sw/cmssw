import FWCore.ParameterSet.Config as cms

AlCaRecoTriggerBitsRcdRead = cms.EDAnalyzer(
    "AlCaRecoTriggerBitsRcdRead",
    pythonOutput = cms.untracked.bool(True) 
    )
