import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    #bool loadAll = true
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBGainsRcd'),
        tag = cms.string('CSCDBGains_ideal')
    ), 
        cms.PSet(
            record = cms.string('CSCDBNoiseMatrixRcd'),
            tag = cms.string('CSCDBNoiseMatrix_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCDBCrosstalkRcd'),
            tag = cms.string('CSCDBCrosstalk_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCDBPedestalsRcd'),
            tag = cms.string('CSCDBPedestals_ideal')
        )),
    #read constants from DB
    connect = cms.string('frontier://FrontierDev/CMS_COND_CSC')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod1 = cms.EDAnalyzer("CSCCrossTalkDBReadAnalyzer")

process.prod2 = cms.EDAnalyzer("CSCGainsDBReadAnalyzer")

process.prod3 = cms.EDAnalyzer("CSCNoiseMatrixDBReadAnalyzer")

process.prod4 = cms.EDAnalyzer("CSCPedestalDBReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1*process.prod2*process.prod3*process.prod4)
process.ep = cms.EndPath(process.output)

