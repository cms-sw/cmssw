# Test read of bad channel db for CSC wires
# This version 17.06.2009 Tim Cox

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
    connect = cms.string('sqlite_file:BadWires_17June2009.db')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

## Set the run number > 100K
process.source = cms.Source("EmptySource",
 firstRun = cms.untracked.uint32(100001)
)

process.analyze = cms.EDAnalyzer("CSCReadBadWiresAnalyzer",
    outputToFile = cms.bool(False),
    readBadChannels = cms.bool(True)
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)

