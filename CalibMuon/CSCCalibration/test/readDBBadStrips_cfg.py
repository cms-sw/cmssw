# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCBadStripsRcd'),
        tag = cms.string('CSCBadStrips_new_popcon')
    )),
    #read from sqlite_file
    connect = cms.string('sqlite_file:BadStrips.db'),
    # read from database
    #string connect="frontier://FrontierDev/CMS_COND_CSC"
    #string connect = "oracle://cms_orcoff_int2r/CMS_COND_CSC"
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("CSCReadBadStripsAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.output)

