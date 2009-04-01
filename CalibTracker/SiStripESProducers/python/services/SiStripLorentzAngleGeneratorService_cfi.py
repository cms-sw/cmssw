import FWCore.ParameterSet.Config as cms


SiStripLorentzAngleGenerator = cms.Service("SiStripLorentzAngleGenerator",
                                           TIB_EstimatedValue = cms.double(0.01784),
                                           TOB_EstimatedValue = cms.double(0.02315),
                                           TIB_PerCent_Err    = cms.double(20),
					   TOB_PerCent_Err    = cms.double(10),
                                           file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),         
                                           )




