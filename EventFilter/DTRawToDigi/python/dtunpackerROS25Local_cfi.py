import FWCore.ParameterSet.Config as cms

#
# configuration for running the DT unpacker on data acquired thorugh ROS25
#
dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('ROS25'),
    inputLabel = cms.InputTag('rawDataCollector'),
    fedbyType = cms.bool(False),
    useStandardFEDid = cms.bool(True),
    dqmOnly = cms.bool(False),                       
    readOutParameters = cms.PSet(
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
)


