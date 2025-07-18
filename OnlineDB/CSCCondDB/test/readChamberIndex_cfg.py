import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
    ),
    #bool loadAll = true
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCChamberIndexRcd'),
        tag = cms.string('CSCChamberIndex')
    )),
    #read constants from DB
    #string connect = "frontier://FrontierDev/CMS_COND_CSC"
    connect = cms.string('sqlite_file:CSCChamberIndexValues_20X.db')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod1 = cms.EDAnalyzer("CSCReadChamberIndexValuesAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1)
process.ep = cms.EndPath(process.output)

