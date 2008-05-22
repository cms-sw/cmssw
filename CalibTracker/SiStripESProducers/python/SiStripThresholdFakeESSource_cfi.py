import FWCore.ParameterSet.Config as cms

SiStripThresholdFakeESSource = cms.ESSource("SiStripThresholdFakeESSource",
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    HighTh = cms.double(5.0),
    LowTh = cms.double(2.0)
)


