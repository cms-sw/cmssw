import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

# name of DQM Source program
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hlxdqmsource = DQMEDAnalyzer('HLXMonitor',
    NBINS = cms.untracked.uint32(335), ## 10 bunch crossings per bin

    Style = cms.untracked.string('BX'), ## BX for bunch crossing vs. Num events

    outputFile = cms.untracked.string('DQM'),
    # 2 random data
    HLXDAQIP = cms.untracked.string('vmepcs2f17-19'),
    subSystemName = cms.untracked.string('HLX'),
    XMIN = cms.untracked.double(100.0),
    XMAX = cms.untracked.double(3450.0),
    SourcePort = cms.untracked.uint32(51001),
    AquireMode = cms.untracked.uint32(0), ## 0 TCP data, 1 constant fake data

    # History for time vs. Num events
    # Dist for Distribution of Num events
    Accumulate = cms.untracked.bool(True),
    outputDir = cms.untracked.string('/cms/mon/data/dqm/lumi/root/dqmsource'),
    SavePeriod = cms.untracked.uint32(64)
)



