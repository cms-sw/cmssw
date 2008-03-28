import FWCore.ParameterSet.Config as cms

# Module for DT data unpacking: produces a DTDigiCollection and - on demand - 
# a DTLocalTriggerCollection
muonDTDigis = cms.EDFilter("DTUnpackingModule",
    dataType = cms.string('DDU'),
    fedColl = cms.untracked.string('source'),
    fedbyType = cms.untracked.bool(False),
    useStandardFEDid = cms.untracked.bool(True),
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)


