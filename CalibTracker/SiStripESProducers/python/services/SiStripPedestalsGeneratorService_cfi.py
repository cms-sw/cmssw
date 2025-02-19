import FWCore.ParameterSet.Config as cms


SiStripPedestalsGenerator = cms.Service("SiStripPedestalsGenerator",
                                        printDebug = cms.untracked.uint32(5),
                                        PedestalsValue = cms.uint32(30),
                                        file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                        )




