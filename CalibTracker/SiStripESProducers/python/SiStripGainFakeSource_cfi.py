import FWCore.ParameterSet.Config as cms

siStripGainFakeESSource = cms.ESSource("SiStripGainFakeESSource",
    appendToDataLabel = cms.string('fakeAPVGain'),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat')
)


