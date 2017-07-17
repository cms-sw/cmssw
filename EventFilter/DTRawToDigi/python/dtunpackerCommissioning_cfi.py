import FWCore.ParameterSet.Config as cms
import EventFilter.DTRawToDigi.dtUnpackingModule_cfi

# Module for DT data unpacking: produces a DTDigiCollection and - on demand - 
# a DTLocalTriggerCollection
# Configuration for Comissioning data
dtunpacker = EventFilter.DTRawToDigi.dtUnpackingModule_cfi.dtUnpackingModule.clone()
dtunpacker.dataType = cms.string('DDU')
dtunpacker.inputLabel = cms.InputTag('rawDataCollector')
dtunpacker.useStandardFEDid = cms.untracked.bool(True)
dtunpacker.dqmOnly = cms.bool(False)
dtunpacker.readOutParameters = cms.PSet(
    debug = cms.untracked.bool(False),
    rosParameters = cms.PSet(
        writeSC = cms.untracked.bool(True),
        readingDDU = cms.untracked.bool(True),
        performDataIntegrityMonitor = cms.untracked.bool(False),
        readDDUIDfromDDU = cms.untracked.bool(True),
        debug = cms.untracked.bool(False),
        localDAQ = cms.untracked.bool(True)
    ),
    localDAQ = cms.untracked.bool(True),
    performDataIntegrityMonitor = cms.untracked.bool(False)
)
