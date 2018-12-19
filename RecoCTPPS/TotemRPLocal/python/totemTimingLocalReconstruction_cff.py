import FWCore.ParameterSet.Config as cms

from CondFormats.CTPPSReadoutObjects.timingCalibrationESSource_cfi import timingCalibrationESSource
timingCalibrationESSource.calibrationFile = cms.FileInPath('RecoCTPPS/TotemRPLocal/data/timing_offsets_ufsd_2018.dec18.cal.json')

# reco hit production
from RecoCTPPS.TotemRPLocal.totemTimingRecHits_cfi import totemTimingRecHits

# local track fitting
from RecoCTPPS.TotemRPLocal.totemTimingLocalTracks_cfi import totemTimingLocalTracks

totemTimingLocalReconstruction = cms.Sequence(
    totemTimingRecHits *
    totemTimingLocalTracks
)

