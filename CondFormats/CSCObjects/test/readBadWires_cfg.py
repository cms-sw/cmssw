import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    timetype = cms.string('runnumber'),
    # For testing, sqlite database file expected to be in local directory 
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCBadWiresRcd'),
        tag = cms.string('CSCBadWires')
    )),
    connect = cms.string('sqlite_file:BadWires_11June2009.db')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.analyze = cms.EDAnalyzer("CSCReadBadWiresAnalyzer",
    outputToFile = cms.bool(False),
    readBadChannels = cms.bool(True)
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)

