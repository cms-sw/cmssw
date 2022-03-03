import FWCore.ParameterSet.Config as cms

from CalibPPS.TimingCalibration.ppsTimingCalibrationPCLWorker_cfi import ppsTimingCalibrationPCLWorker

MEtoEDMConvertPPSTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSTimingCalibrationPCL'),
    deleteAfterCopy = cms.untracked.bool(True),
)

ppsTimingCalibrationPCLWorker.diamondRecHitTags=cms.VInputTag(cms.InputTag("ctppsDiamondRecHitsAlCaRecoProducer"),
                                         			  cms.InputTag("ctppsDiamondRecHits"))

taskALCARECOPromptCalibProdPPSTimingCalib = cms.Task(
    ppsTimingCalibrationPCLWorker,
    MEtoEDMConvertPPSTimingCalib
)
