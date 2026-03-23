import FWCore.ParameterSet.Config as cms

from ..modules.hltPixelTracksSoAMonitorCPU_cfi import *
from ..modules.hltPixelTracksSoAMonitorGPU_cfi import *
from ..modules.hltPixelTracksSoACompareGPUvsCPU_cfi import *
from ..modules.hltPixelTrackToTrackSerialSync_cfi import *
from ..modules.hltInitialStepSeedsTrackToTrackSerialSync_cfi import *

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTDQMTrackReconstruction = cms.Sequence()

from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTDQMTrackReconstruction,
    cms.Sequence(
        hltPixelTracksSoAMonitorCPU +
        hltPixelTracksSoAMonitorGPU +
        hltPixelTracksSoACompareGPUvsCPU +
        hltPixelTrackToTrackSerialSync +
        hltInitialStepSeedsTrackToTrackSerialSync
    )
)
