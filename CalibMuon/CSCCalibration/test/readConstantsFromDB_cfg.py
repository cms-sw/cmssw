import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCGainsRcd'),
        tag = cms.string('CSCGains_ideal')
    ), 
        cms.PSet(
            record = cms.string('CSCNoiseMatrixRcd'),
            tag = cms.string('CSCNoiseMatrix_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCcrosstalkRcd'),
            tag = cms.string('CSCCrosstalk_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCPedestalsRcd'),
            tag = cms.string('CSCPedestals_ideal')
        )),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    catalog = cms.untracked.string('relationalcatalog_oracle://cms_orcoff_int2r/CMS_COND_GENERAL'), ##cms_orcoff_int2r/CMS_COND_GENERAL"

    timetype = cms.string('runnumber'),
    #read constants from DB
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_CSC')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod1 = cms.EDAnalyzer("CSCCrossTalkReadAnalyzer")

process.prod2 = cms.EDAnalyzer("CSCGainsReadAnalyzer")

process.prod3 = cms.EDAnalyzer("CSCNoiseMatrixReadAnalyzer")

process.prod4 = cms.EDAnalyzer("CSCPedestalReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1*process.prod2*process.prod3*process.prod4)
process.ep = cms.EndPath(process.output)


