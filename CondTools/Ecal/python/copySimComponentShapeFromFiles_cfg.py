import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.producedEcalSimComponentShape = cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.getSimComponentShapeFromFile = cms.untracked.bool(True) ### if set False hadrdcoded shapes will be loaded by default


### phase II Pulse Shapes
process.EcalTrivialConditionRetriever.sim_component_shape_TI = cms.untracked.double(1)
process.EcalTrivialConditionRetriever.sim_component_shape_EB_thresh = cms.double(0.00013)
fileNames = [f"EB_SimComponentShape_PhaseI_depth{i}.txt" for i in range(0,23)]
#fileNames = [f"EB_SimComponentShape_PhaseII_depth{i}.txt" for i in range(0,23)]
process.EcalTrivialConditionRetriever.EBSimComponentShapeFiles = cms.untracked.vstring(fileNames)


process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_31X_ECAL'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:EBSimComponentShape_PhaseI.db'
#process.CondDB.connect = 'sqlite_file:EBSimComponentShape_PhaseII.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
  firstValue = cms.uint64(1),
  lastValue = cms.uint64(1),
  timetype = cms.string('runnumber'),
  interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalSimComponentShapeRcd'),
      tag = cms.string('EcalSimComponentShape_PhaseI')
      #tag = cms.string('EcalSimComponentShape_PhaseII')
    )
  )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
  timetype = cms.string('runnumber'),
  toCopy = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalSimComponentShapeRcd'),
      container = cms.string('EcalSimComponentShape')
    )
  )
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)
