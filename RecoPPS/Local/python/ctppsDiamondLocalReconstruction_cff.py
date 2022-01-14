import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.ctppsDiamondRecHits_cfi import ctppsDiamondRecHits

# local track fitting
from RecoPPS.Local.ctppsDiamondLocalTracks_cfi import ctppsDiamondLocalTracks

from CalibPPS.ESProducers.ppsTimingCalibrationLUTESSource_cfi import ppsTimingCalibrationLUTESSource

ppsTimingCalibrationLUTESSource.calibrationFile = cms.FileInPath('RecoPPS/Local/data/LUT_cal_test.json')
ctppsDiamondRecHits.timingCalibrationLUTTag=cms.string('ppsTimingCalibrationLUTESSource:')

ctppsDiamondLocalReconstructionTask = cms.Task(
    ctppsDiamondRecHits,
    ctppsDiamondLocalTracks
)
ctppsDiamondLocalReconstruction = cms.Sequence(ctppsDiamondLocalReconstructionTask)
