import FWCore.ParameterSet.Config as cms

from RecoPPS.Configuration.recoCTPPS_cff import *
from CalibPPS.TimingCalibration.ppsTimingCalibrationPCLWorker_cfi import ppsTimingCalibrationPCLWorker

MEtoEDMConvertPPSTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSTimingCalibrationPCL'),
    deleteAfterCopy = cms.untracked.bool(True),
)

# uncalibrated rechits/tracks
ctppsDiamondRecHits.applyCalibration = False

# calibrated rechits/tracks
ctppsDiamondCalibRecHits = ctppsDiamondRecHits.clone(
    applyCalibration = True
)
ctppsDiamondCalibLocalTracks = ctppsDiamondLocalTracks.clone(
    recHitsTag = cms.InputTag('ctppsDiamondCalibRecHits')
)
recoDiamondCalibLocalReconstructionTask = cms.Task(
    ctppsDiamondCalibRecHits,
    ctppsDiamondCalibLocalTracks
)
recoDiamondCalib = cms.Sequence(recoDiamondCalibLocalReconstructionTask)

seqALCARECOPPSTimingCalib = cms.Sequence(
    recoDiamondCalib *
    ppsTimingCalibrationPCLWorker *
    MEtoEDMConvertPPSTimingCalib
)
