import FWCore.ParameterSet.Config as cms

process = cms.Process("fedBadChannelFromNoiseRun")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
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
process.load("CalibTracker.SiStripESProducers.fake.SiStripBadModuleFedErrFakeESSource_cfi")
process.siStripBadModuleFedErrFakeESSource.appendToDataLabel = cms.string('BadModules_from_FEDBadChannel')
process.SiStripBadModuleFedErrService.ReadFromFile = cms.bool(False)

process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
        cms.PSet(record = cms.string('SiStripBadModuleFedErrRcd'), tag = cms.string('BadModules_from_FEDBadChannel'))
#        cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string(''))
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)
process.siStripQualityESProducer.ThresholdForReducedGranularity = cms.double(0.3)

#### Add these lines to produce a tracker map
process.load("DQMServices.Core.DQMStore_cfg")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
####

process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
    dataLabel = cms.untracked.string('')
)


process.p = cms.Path(process.stat)

