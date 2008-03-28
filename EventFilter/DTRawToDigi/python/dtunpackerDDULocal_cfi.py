import FWCore.ParameterSet.Config as cms

# Module for DT data unpacking: produces a DTDigiCollection and - on demand - 
# a DTLocalTriggerCollection
# Configuration for Local DAQ at MTCC
dtunpacker = cms.EDFilter("DTUnpackingModule",
    dataType = cms.string('DDU'),
    useStandardFEDid = cms.untracked.bool(True),
    fedbyType = cms.untracked.bool(True),
    readOutParameters = cms.PSet(
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
)


