import FWCore.ParameterSet.Config as cms

from RecoPPS.Configuration.recoCTPPS_cff import ctppsDiamondRecHits as _ctppsDiamondRecHits
from CalibPPS.TimingCalibration.ppsTimingCalibrationPCLWorker_cfi import ppsTimingCalibrationPCLWorker

MEtoEDMConvertPPSTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSTimingCalibrationPCL'),
    deleteAfterCopy = cms.untracked.bool(True),
)

# calibrated rechits/tracks
ctppsDiamondUncalibRecHits = _ctppsDiamondRecHits.clone(
    applyCalibration = False
)
recoDiamondUncalibLocalReconstructionTask = cms.Task(
    ctppsDiamondUncalibRecHits,
)
# this sequence will be updated to include tracking based on the last
# calibration values to extract per-channel timing precision estimation
recoDiamond = cms.Sequence(recoDiamondUncalibLocalReconstructionTask)

taskALCARECOPPSTimingCalib = cms.Task(
    recoDiamond *
    ppsTimingCalibrationPCLWorker *
    MEtoEDMConvertPPSTimingCalib
)
