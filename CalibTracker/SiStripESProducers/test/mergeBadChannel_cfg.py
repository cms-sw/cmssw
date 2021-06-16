import FWCore.ParameterSet.Config as cms

process = cms.Process("BadChannelMerge")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')


process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(258714),
    lastValue = cms.uint64(258714),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.load('DQMServices.Core.DQMStore_cfi')

process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
       cms.PSet(record = cms.string('SiStripBadFiberRcd'), tag = cms.string('')),
       cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string(''))
       ## BadChannel list from FED errors is added below
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)
process.siStripQualityESProducer.ThresholdForReducedGranularity = cms.double(0.3)

#### Add these lines to produce a tracker map
process.load("DQM.SiStripCommon.TkHistoMap_cff")
####

from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        BadComponentsFromFedErrors=siStripQualityStatistics.clone(
            Add=cms.bool(True)
            )
        )

process.p = cms.Path(process.stat)

