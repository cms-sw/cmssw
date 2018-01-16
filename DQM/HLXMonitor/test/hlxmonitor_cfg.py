# The following comments couldn't be translated into the new config version:

#service = DQMShipMonitoring{
# // event-period for shipping monitoring to collector (default: 25)
# untracked uint32 period = 1
#}

import FWCore.ParameterSet.Config as cms

process = cms.Process("HLXDQM")
# back-end interface service
process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(300000)
)
process.source = cms.Source("EmptySource")

process.hlxdqmsource = DQMStep1Module('HLXMonitor',
    NBINS = cms.untracked.uint32(297),
    Style = cms.untracked.string('BX'), ## BX for bunch crossing vs. Num events

    outputFile = cms.untracked.string('DQMOut'),
    # History for time vs. Num events
    # Dist for Distribution of Num events
    Accumulate = cms.untracked.bool(True),
    XMAX = cms.untracked.double(3564.0),
    SourcePort = cms.untracked.uint32(51001),
    AquireMode = cms.untracked.uint32(0), ## 0 TCP data, 1 constant fake data

    HLXIP = cms.untracked.string('vmepcs2f17-19'),
    outputDir = cms.untracked.string('/cms/mon/data/dqm/lumi/root/dqmsource'),
    SavePeriod = cms.untracked.uint32(64)
)

process.p = cms.Path(process.hlxdqmsource)


