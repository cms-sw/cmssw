import FWCore.ParameterSet.Config as cms

# DQM Environment
dqmEnv = cms.EDAnalyzer("DQMEventInfo",
    # put your subsystem name here (this goes into the foldername)
    subSystemFolder = cms.untracked.string('YourSubsystem'),
    # set the window for eventrate calculation (in minutes)
    eventRateWindow = cms.untracked.double(0.5),
    # define folder to store event info (default: EventInfo)
    eventInfoFolder = cms.untracked.string('EventInfo')
)

