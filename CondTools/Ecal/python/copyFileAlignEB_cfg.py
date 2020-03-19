import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.getEBAlignmentFromFile = cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.EBAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data_test/EBAlignment_2018.txt')
process.EcalTrivialConditionRetriever.getEEAlignmentFromFile = cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.EEAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data_test/EEAlignment_2018.txt')
process.EcalTrivialConditionRetriever.getESAlignmentFromFile = cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.ESAlignmentFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data_test/ESAlignment_2018.txt')

#process.load("EcalTrivialAlignment_cfi")

process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
#process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDB.connect = 'sqlite_file:EBAlignment_test.db'

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
       cms.PSet(
          record = cms.string('EBAlignmentRcd'),
          tag = cms.string('EBAlignment_test')
       )
    )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet( 
       cms.PSet(
          record = cms.string('EBAlignmentRcd'),
          container = cms.string('EBAlignment')
       )
    )
)


process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)

