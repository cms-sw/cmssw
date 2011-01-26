import FWCore.ParameterSet.Config as cms

SiStripLatencyGenerator = cms.Service("SiStripLatencyGenerator",
                                     # printDebug = cms.untracked.uint32(5),
                                     file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                     latency = cms.uint32(1),
                                     mode = cms.uint32(37)
                                     )


