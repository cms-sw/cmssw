import FWCore.ParameterSet.Config as cms

siStripThresholdFakeOnDB = cms.EDFilter("SiStripThresholdFakeOnDB",
    #cards relevant to mother class
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    doStoreOnDB = cms.bool(True),
    Record = cms.string('SiStripThresholdRcd')
)


