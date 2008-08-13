import FWCore.ParameterSet.Config as cms

DTDataIntegrityTask = cms.Service("DTDataIntegrityTask",
    outputFile = cms.untracked.string('DataIntegrity.root'),
    writeHisto = cms.untracked.bool(False),
    enableMonitorDaemon = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),
    timeBoxLowerBound = cms.untracked.int32(0),
    timeBoxUpperBound = cms.untracked.int32(10000)
)


