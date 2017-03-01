import FWCore.ParameterSet.Config as cms
import EventFilter.DTRawToDigi.dtUnpackingModule_cfi

#
# configuration for running the DT unpacker on data acquired thorugh ROS25
#
dtunpacker =  EventFilter.DTRawToDigi.dtUnpackingModule_cfi.dtUnpackingModule.clone()
dtunpacker.dataType = cms.string('ROS25')
dtunpacker.inputLabel = cms.InputTag('rawDataCollector')
dtunpacker.useStandardFEDid = cms.bool(True)
dtunpacker.dqmOnly = cms.bool(False)
dtunpacker. readOutParameters = cms.PSet(
    debug = cms.untracked.bool(False),
    rosParameters = cms.PSet(
        writeSC = cms.untracked.bool(True),
        readingDDU = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False),
        readDDUIDfromDDU = cms.untracked.bool(False),
        debug = cms.untracked.bool(False),
        localDAQ = cms.untracked.bool(True)
    ),
    localDAQ = cms.untracked.bool(True),
    performDataIntegrityMonitor = cms.untracked.bool(False)
)
