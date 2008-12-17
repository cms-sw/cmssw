import FWCore.ParameterSet.Config as cms

#
# configuration for running the DT unpacker on data acquired thorugh ROS25
#
dtunpacker = cms.EDFilter("DTUnpackingModule",
    dataType = cms.string('ROS25'),
    inputLabel = cms.untracked.InputTag('source'),
    fedbyType = cms.untracked.bool(False),
    useStandardFEDid = cms.untracked.bool(True),
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(True),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(False),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(False),
            debug = cms.untracked.bool(True),
            localDAQ = cms.untracked.bool(True)
        ),
        localDAQ = cms.untracked.bool(True),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)


