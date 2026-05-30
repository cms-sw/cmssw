import FWCore.ParameterSet.Config as cms

from ..modules.hltHGCALGPUvsCPUComparison_cfi import *

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTDQMHGCALReconstruction = cms.Sequence()

from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTDQMHGCALReconstruction,
    cms.Sequence(
        hltHGCALGPUvsCPUComparisonHists
    )
)
