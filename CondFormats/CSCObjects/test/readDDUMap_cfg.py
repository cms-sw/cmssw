# The following comments couldn't be translated into the new config version:

#read constants from DB

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    #bool loadAll = true
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDDUMapRcd'),
        tag = cms.string('CSCDDUMap')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_CSC')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod1 = cms.EDAnalyzer("CSCReadDDUMapValuesAnalyzer")

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1)
process.ep = cms.EndPath(process.printEventNumber)

