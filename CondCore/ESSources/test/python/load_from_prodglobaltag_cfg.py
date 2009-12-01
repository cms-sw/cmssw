import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT09_R_V9::All'
#'GR09_P_V6::All'
#'MC_31X_V9::All'
#'GR09_31X_V5P::All'
process.GlobalTag.DumpStat =  True
process.GlobalTag.pfnPrefix = "frontier://FrontierArc/"
process.GlobalTag.pfnPostfix = "_0911"

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



