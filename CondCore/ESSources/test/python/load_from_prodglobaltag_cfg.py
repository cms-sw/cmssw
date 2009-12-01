import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)


process.get = cms.EDFilter("EventSetupRecordDataGetter",
                           toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('RunInfoRcd'),
    data = cms.vstring('RunInfo')
    )
    ),
                           verbose = cms.untracked.bool(True)
                           )

process.p = cms.Path(process.get)



