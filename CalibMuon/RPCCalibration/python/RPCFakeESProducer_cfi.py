import FWCore.ParameterSet.Config as cms

RPCFakeCalibration = cms.ESSource("RPCFakeCalibration",
    effmapfile = cms.FileInPath('CalibMuon/RPCCalibration/data/RPCDetId_Eff.dat'),
    noisemapfile = cms.FileInPath('CalibMuon/RPCCalibration/data/RPCDetId_Noise.dat'),
    timingMap = cms.FileInPath('CalibMuon/RPCCalibration/data/RPCTiming.dat'),
    clsmapfile = cms.FileInPath('CalibMuon/RPCCalibration/data/ClSizeTot.dat')
)



