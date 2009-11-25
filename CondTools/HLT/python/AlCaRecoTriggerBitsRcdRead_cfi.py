import FWCore.ParameterSet.Config as cms

AlCaRecoTriggerBitsRcdRead = cms.EDAnalyzer(
    "AlCaRecoTriggerBitsRcdRead",
    outputType = cms.untracked.string("twiki") # or text, python (future: html?) 
    )
