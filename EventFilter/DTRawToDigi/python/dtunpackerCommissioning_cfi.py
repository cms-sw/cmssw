import FWCore.ParameterSet.Config as cms

# Module for DT data unpacking: produces a DTDigiCollection and - on demand - 
# a DTLocalTriggerCollection
# Configuration for Comissioning data
dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('rawDataCollector'),
    fedbyType = cms.bool(False),
    useStandardFEDid = cms.untracked.bool(True),
    dqmOnly = cms.bool(False),                       
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


