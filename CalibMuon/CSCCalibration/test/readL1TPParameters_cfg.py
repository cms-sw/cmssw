# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

# read from database
#string connect="frontier://FrontierDev/CMS_COND_CSC"

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    #read from sqlite_file
    #string connect = "sqlite_file:DBL1TPParameters.db"
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCL1TPParametersRcd'),
        tag = cms.string('CSCL1TPParameters')
    )),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_CSC'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("CSCL1TPParametersReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.output)


