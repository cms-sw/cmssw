import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *
from RecoCTPPS.Configuration.recoCTPPS_cff import *
from CalibPPS.TimingCalibration.ppsTimingCalibrationPCLWorker_cfi import ppsTimingCalibrationPCLWorker

ALCARECOPPSTimingCalib = ppsTimingCalibrationPCLWorker.clone()

MEtoEDMConvertPPSTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSTimingCalibrationPCL'),
    deleteAfterCopy = cms.untracked.bool(True),
)

seqALCARECOPPSTimingCalib = cms.Sequence(
    ctppsRawToDigi *
    recoCTPPS *
    ALCARECOPPSTimingCalib *
    MEtoEDMConvertPPSTimingCalib
)
