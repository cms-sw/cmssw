import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1),
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      DBParameters = cms.PSet(
                                          messageLevel = cms.untracked.int32(2),
                                          authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                      ),
                                      toGet = cms.VPSet(cms.PSet(
                                          record = cms.string('SiStripBadStripRcd'),
                                          tag = cms.string('SiStripBadChannel_v1')
                                      )),
                                      connect = cms.string('sqlite_file:SiStripConditionsDBFile.db')
                                      )

process.prod = cms.EDAnalyzer("SiStripBadStripReader",
    printDebug = cms.untracked.bool(True)
)

process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.print)


