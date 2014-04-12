import FWCore.ParameterSet.Config as cms

SiStripHashedDetIdFakeESProducer = cms.ESSource("SiStripHashedDetIdFakeESProducer",
    DetIds = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat')
)


