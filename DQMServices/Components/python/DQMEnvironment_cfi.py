import FWCore.ParameterSet.Config as cms

# DQM Environment
dqmEnv = cms.EDFilter("DQMEventInfo",
    # put your subsystem name here (this goes into the foldername)
    subSystemFolder = cms.untracked.string('YourSubsystem'),
    # set the window for eventrate calculation (in minutes)
    eventRateWindow = cms.untracked.double(0.5),
    # define folder to store event info (default: EventInfo)
    eventInfoFolder = cms.untracked.string('EventInfo')
)

# DQM file saver module
dqmSaver = cms.EDFilter("DQMFileSaver",
    # Name of the producer.
    producer = cms.untracked.string('DQM'),
    # Save file every N events (-1: disabled)
    saveByEvent = cms.untracked.int32(-1),
    # Ignore run number for MC data (-1: disabled)
    forceRunNumber = cms.untracked.int32(-1),
    # Save file every N runs (-1: disabled)
    saveByRun = cms.untracked.int32(1),
    # Name of the processing workflow.
    workflow = cms.untracked.string(''),
    # Save file at the end of the job
    saveAtJobEnd = cms.untracked.bool(False),
    # Possible conventions are "Online", "Offline" and "RelVal".
    convention = cms.untracked.string('Offline'),
    # Save file every N minutes (-1: disabled)
    saveByMinute = cms.untracked.int32(-1),
    # Directory in which to save the files.
    dirName = cms.untracked.string('.'),
    # Save file every N lumi sections (-1: disabled)
    saveByLumiSection = cms.untracked.int32(-1)
)


