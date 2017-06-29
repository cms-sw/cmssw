import FWCore.ParameterSet.Config as cms

clusterProd = cms.EDProducer("CTPPSPixelClusterProducer",
                                     label=cms.untracked.string("ctppsPixelDigis"),
                                     RPixVerbosity = cms.int32(0),
                                     SeedADCThreshold = cms.int32(10),
                                     ADCThreshold = cms.int32(10),
                                     ElectronADCGain = cms.double(135.0),
                                     VCaltoElectronOffset = cms.int32(-411),
                                     VCaltoElectronGain = cms.int32(50),
#                                     CalibrationFile = cms.string("file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/Gain_Fed_1462-1463_Run_107.root"),
                                     doSingleCalibration = cms.bool(False)
)
