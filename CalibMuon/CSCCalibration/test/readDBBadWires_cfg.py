# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCBadWiresRcd'),
        tag = cms.string('CSCBadWires_2009_mc')
    )),
    #read from sqlite_file
    connect = cms.string('sqlite_file:CSCBadWires_2009_mc.db'),
    #connect = cms.string('frontier://FrontierProd/CMS_COND_31X_CSC'),
    # read from database
    #string connect="frontier://FrontierDev/CMS_COND_CSC"
    #string connect = "oracle://cms_orcoff_int2r/CMS_COND_CSC"
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1)
    )

process.analyze = cms.EDAnalyzer("CSCReadBadWiresAnalyzer",
                              outputToFile = cms.bool(True),                           
                              readBadChannels = cms.bool(True),
                              me42installed = cms.bool(True)
)
process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.output)

