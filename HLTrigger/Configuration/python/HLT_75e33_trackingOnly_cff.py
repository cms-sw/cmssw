import FWCore.ParameterSet.Config as cms

from .HLT_75e33_cff import fragment

fragment.load("HLTrigger/Configuration/HLT_75e33/paths/MC_TRK_cfi")

fragment.schedule = cms.Schedule(*[
    fragment.MC_TRK,
    fragment.HLTriggerFinalPath,
    fragment.HLTAnalyzerEndpath,
])
