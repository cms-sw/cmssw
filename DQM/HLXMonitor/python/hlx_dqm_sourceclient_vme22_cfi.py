import FWCore.ParameterSet.Config as cms

# name of DQM Source program
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hlxdqmsource = DQMEDAnalyzer('HLXMonitor',

    Style = cms.untracked.string('BX'), ## BX for bunch crossing vs. Num events

    outputFile = cms.untracked.string('DQM_V0001'),
    # 2 random data
    PrimaryHLXDAQIP = cms.untracked.string('vmepcs2f17-18'),
    SecondaryHLXDAQIP = cms.untracked.string('vmepcs2f17-19'),
    subSystemName = cms.untracked.string('HLX'),
    NBINS = cms.untracked.uint32(289), ## 12 bunch crossings per bin
    XMIN = cms.untracked.double(0.0),
    XMAX = cms.untracked.double(3468.0),
    maximumNumLS = cms.untracked.uint32(1850),
    SourcePort = cms.untracked.uint32(51001),
    AquireMode = cms.untracked.uint32(0), ## 0 TCP data, 1 constant fake data
    #AquireMode = cms.untracked.uint32(1), ## 0 TCP data, 1 constant fake data
    ReconnectionTime = cms.untracked.uint32(60),
    MinLSBeforeSave = cms.untracked.uint32(4),

    ## For private DQM set this to normal value
    ## since it is not set by dqmEnv. Default is EventInfoHLX
    eventInfoFolderHLX = cms.untracked.string('EventInfo'),

    # History for time vs. Num events
    # Dist for Distribution of Num events
    Accumulate = cms.untracked.bool(True),
    outputDir = cms.untracked.string('/opt/dqm/data/live'),
    SavePeriod = cms.untracked.uint32(64)
)



