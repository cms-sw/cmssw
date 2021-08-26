import FWCore.ParameterSet.Config as cms

from RecoPPS.Configuration.recoCTPPS_cff import diamondSampicLocalReconstruction
from CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLWorker_cfi import PPSDiamondSampicTimingCalibrationPCLWorker
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

diamondSampicCalibrationDQM = DQMEDAnalyzer('DiamondSampicCalibrationDQMSource',
    tagRecHits = cms.InputTag("totemTimingRecHits"),
    verbosity = cms.untracked.uint32(0)
)

# load DQM framework
from DQM.Integration.config.environment_cfi import *
dqmEnv.subSystemFolder = "CTPPS"
dqmEnv.eventInfoFolder = "EventInfo"
dqmSaver.path = ""
dqmSaver.tag = "CTPPS"

MEtoEDMConvertPPSDiamondSampicTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSDiamondSampicTimingCalibrationPCL'),
    deleteAfterCopy = cms.untracked.bool(True),
)

seqALCARECOPPSDiamondSampicTimingCalib = cms.Sequence(
    diamondSampicLocalReconstruction*
    PPSDiamondSampicTimingCalibrationPCLWorker*
    diamondSampicCalibrationDQM*
    dqmEnv*
    dqmSaver*
    MEtoEDMConvertPPSDiamondSampicTimingCalib
)
