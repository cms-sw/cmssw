# Test bad chambers data base from Oana, Tim Cox 21.10.2008/28.01.2009

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
        record = cms.string('CSCBadChambersRcd'),
        tag = cms.string('CSCBadChambers_ME42')
    )),
    connect = cms.string('sqlite_file:DBBadChambers.db')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.analyze = cms.EDAnalyzer("CSCReadBadChambersAnalyzer",
    outputToFile = cms.bool(False),
    readBadChambers = cms.bool(True),
    me42installed = cms.bool(True)                                 
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)

