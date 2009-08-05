import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

# name of DQM Source program
hlxdqmsource = cms.EDFilter("HLXMonitor",

    Style = cms.untracked.string('BX'), ## BX for bunch crossing vs. Num events

    outputFile = cms.untracked.string('DQM_V0001'),
    # 2 random data
    NewRun_Reset = cms.untracked.bool(True),
    PrimaryHLXDAQIP = cms.untracked.string('vmepcs2f17-18'),
    SecondaryHLXDAQIP = cms.untracked.string('vmepcs2f17-19'),
    subSystemName = cms.untracked.string('HLX'),
    XMIN = cms.untracked.double(100.0),
    XMAX = cms.untracked.double(3460.0),
    NBINS = cms.untracked.uint32(280),
    SourcePort = cms.untracked.uint32(51001),
    AquireMode = cms.untracked.uint32(0), ## 0 TCP data, 1 constant fake data
    ReconnectionTime = cms.untracked.uint32(60), ## re-try every minute

    # History for time vs. Num events
    # Dist for Distribution of Num events
    Accumulate = cms.untracked.bool(True),
    outputDir = cms.untracked.string('/cms/mon/data/dqm/lumi/root/dqmsource'),
    SavePeriod = cms.untracked.uint32(64)
)



