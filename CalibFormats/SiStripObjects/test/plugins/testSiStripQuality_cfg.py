import FWCore.ParameterSet.Config as cms

process = cms.Process("testSISTRIPQUALITY")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_P_V43::All'

#this block is there to solve issue related to SiStripDetVOffRcd
process.load("CalibTracker.SiStripESProducers.fake.SiStripDetVOffFakeESSource_cfi")
process.es_prefer_fakeSiStripDetVOff = cms.ESPrefer("SiStripDetVOffFakeESSource","siStripDetVOffFakeESSource")

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(206257),
    lastValue  = cms.uint64(999999) #does not matter
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.testSiStripQuality = cms.EDAnalyzer("testSiStripQuality")
process.p = cms.Path(process.testSiStripQuality)
