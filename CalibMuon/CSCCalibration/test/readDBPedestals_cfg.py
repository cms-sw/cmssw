# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBPedestalsRcd'),
        tag = cms.string('CSCDBPedestals_new_popcon')
    )),
    connect = cms.string('sqlite_file:DBPedestals.db'),
    #string connect = "frontier://FrontierDev/CMS_COND_CSC"
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(0)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("CSCPedestalDBReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.output)

