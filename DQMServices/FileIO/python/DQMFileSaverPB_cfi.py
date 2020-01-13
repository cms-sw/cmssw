import FWCore.ParameterSet.Config as cms

# DQM file saver module
dqmSaver = cms.EDAnalyzer("DQMFileSaverPB",
    # Name of the producer.
    producer = cms.untracked.string('DQM'),
    # Directory in which to save the files.
    path = cms.untracked.string('./'),

    # Tag, used in the filename as the third term.
    tag = cms.untracked.string('UNKNOWN'),

    # Control reference saving (default / skip / qtests / all)
    referenceHandling = cms.untracked.string('all'),
    # Control which references are saved for qtests (default: STATUS_OK)
    referenceRequireStatus = cms.untracked.int32(100),

    # If set, EvFDaqDirector is emulated and not used
    fakeFilterUnitMode = cms.untracked.bool(False),
    # Label of the stream
    streamLabel = cms.untracked.string("streamDQMHistograms"),
)
