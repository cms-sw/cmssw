import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoCTPPS.TotemRPLocal.totemTimingRecHits_cfi import totemTimingRecHits
totemTimingRecHits.saturationLimit = cms.double(0.85)
totemTimingRecHits.cfdFraction = cms.double(0.3)
totemTimingRecHits.smoothingPoints = cms.int32(20)
totemTimingRecHits.lowPassFrequency = cms.double(0.7)
totemTimingRecHits.hysteresis = cms.double(5e-3)
totemTimingRecHits.calibrationFile = cms.FileInPath('RecoCTPPS/TotemRPLocal/data/run2_timing_offsets_cal_aug18.json')
totemTimingRecHits.mergeTimePeaks = cms.bool(True);

# local track fitting
from RecoCTPPS.TotemRPLocal.totemTimingLocalTracks_cfi import totemTimingLocalTracks
totemTimingLocalTracks.tolerance = cms.double(0.1);
totemTimingLocalTracks.maxPlaneActiveChannels = cms.int32(2);

totemTimingLocalReconstruction = cms.Sequence(
    totemTimingRecHits *
    totemTimingLocalTracks
)
