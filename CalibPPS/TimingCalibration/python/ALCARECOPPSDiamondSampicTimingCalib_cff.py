import FWCore.ParameterSet.Config as cms

from RecoPPS.Configuration.recoCTPPS_cff import diamondSampicLocalReconstructionTask
from CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLWorker_cfi import PPSDiamondSampicTimingCalibrationPCLWorker
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

MEtoEDMConvertPPSDiamondSampicTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSDiamondSampicTimingCalibrationPCL')
)

taskALCARECOPPSDiamondSampicTimingCalib = cms.Task(
    diamondSampicLocalReconstructionTask,
    PPSDiamondSampicTimingCalibrationPCLWorker,
    MEtoEDMConvertPPSDiamondSampicTimingCalib
)
