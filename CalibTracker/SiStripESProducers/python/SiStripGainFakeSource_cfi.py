import FWCore.ParameterSet.Config as cms

SiStripGainFakeESSource = cms.ESSource("SiStripGainFakeESSource",
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat')
)


