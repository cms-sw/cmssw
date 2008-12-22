import FWCore.ParameterSet.Config as cms


SiStripLorentzAngleGenerator = cms.Service("SiStripLorentzAngleGenerator",
                                           TIB_EstimatedValue = cms.double(0.024),
                                           TOB_EstimatedValue = cms.double(0.030),
                                           PerCent_Err	      = cms.double(20),
                                           file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),         
                                           )




