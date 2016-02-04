import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCCrateMapRcd'),
        tag = cms.string('CSCCrateMap')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_CSC')

)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.action = cms.EDAnalyzer("CSCReadoutMapTest")

process.printEvent = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.action)
process.ep = cms.EndPath(process.printEvent)

