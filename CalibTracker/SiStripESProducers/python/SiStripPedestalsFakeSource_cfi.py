import FWCore.ParameterSet.Config as cms

SiStripPedestalsFakeESSource = cms.ESSource("SiStripPedestalsFakeESSource",
    printDebug = cms.untracked.bool(False),
    HighThValue = cms.double(5.0),
    PedestalsValue = cms.uint32(30),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    LowThValue = cms.double(2.0)
)


