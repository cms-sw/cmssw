# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    #read from database
    #string connect = "frontier://FrontierDev/CMS_COND_CSC"
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBNoiseMatrixRcd'),
        tag = cms.string('CSCDBNoiseMatrix_express')
    )),
    #read from sqlite file
    connect=cms.string("oracle://cms_orcon_prod/CMS_COND_31X_CSC"),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb/'),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("CSCNoiseMatrixDBReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.output)

