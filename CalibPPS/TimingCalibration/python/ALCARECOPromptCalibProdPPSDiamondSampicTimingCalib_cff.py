import FWCore.ParameterSet.Config as cms

from CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLWorker_cfi import PPSDiamondSampicTimingCalibrationPCLWorker
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

MEtoEDMConvertPPSDiamondSampicTimingCalib = cms.EDProducer('MEtoEDMConverter',
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string('AlCaReco/PPSDiamondSampicTimingCalibrationPCL')
)
    
PPSDiamondSampicTimingCalibrationPCLWorker.totemTimingDigiTags=cms.VInputTag(cms.InputTag("totemTimingRawToDigiAlCaRecoProducer","TotemTiming"),
                                         			  cms.InputTag("totemTimingRawToDigi","TotemTiming"))
                                         			  
PPSDiamondSampicTimingCalibrationPCLWorker.totemTimingRecHitTags=cms.VInputTag(cms.InputTag("totemTimingRecHitsAlCaRecoProducer"),
                                                  		    cms.InputTag("totemTimingRecHits"))

taskALCARECOPromptCalibProdPPSDiamondSampic = cms.Task(
    PPSDiamondSampicTimingCalibrationPCLWorker,
    MEtoEDMConvertPPSDiamondSampicTimingCalib
)
