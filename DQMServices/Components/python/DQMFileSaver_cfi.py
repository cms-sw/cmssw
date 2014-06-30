import FWCore.ParameterSet.Config as cms

# DQM file saver module
dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    # Possible conventions are "Online", "Offline" and "RelVal".
    convention = cms.untracked.string('Offline'),
    # Save files in plain ROOT or encode ROOT objects in ProtocolBuffer
    fileFormat = cms.untracked.string('ROOT'),
    # Name of the producer.
    producer = cms.untracked.string('DQM'),
    # Name of the processing workflow.
    workflow = cms.untracked.string(''),
    # Directory in which to save the files.
    dirName = cms.untracked.string('.'),
    # Only save this directory
    filterName = cms.untracked.string(''),
    # Version name to be used in file name.
    version = cms.untracked.int32(1),
    # runIsComplete
    runIsComplete = cms.untracked.bool(False),

    # Save file every N lumi sections (-1: disabled)
    saveByLumiSection = cms.untracked.int32(-1),
    # Save file every N runs (-1: disabled)
    saveByRun = cms.untracked.int32(-1),
    # Save file at the end of the job
    saveAtJobEnd = cms.untracked.bool(True),

    # Ignore run number for MC data (-1: disabled)
    forceRunNumber = cms.untracked.int32(-1),

    # Control reference saving (default / skip / qtests / all)
    referenceHandling = cms.untracked.string('all'),
    # Control which references are saved for qtests (default: STATUS_OK)
    referenceRequireStatus = cms.untracked.int32(100)
)
