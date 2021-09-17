import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.producedEcalSimPulseShape = cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.getSimPulseShapeFromFile = cms.untracked.bool(True) ### if set False hadrdcoded shapes will be loaded by default


### phase I Pulse Shapes
#process.EcalTrivialConditionRetriever.sim_pulse_shape_TI = cms.untracked.double( 1.0)
#process.EcalTrivialConditionRetriever.sim_pulse_shape_EB_thresh = cms.double(0.00013)
#process.EcalTrivialConditionRetriever.sim_pulse_shape_EE_thresh = cms.double(0.00025)
#process.EcalTrivialConditionRetriever.sim_pulse_shape_APD_thresh = cms.double(0)
#process.EcalTrivialConditionRetriever.EBSimPulseShapeFile = cms.untracked.string("EB_SimPulseShape.txt")
#process.EcalTrivialConditionRetriever.EESimPulseShapeFile = cms.untracked.string("EE_SimPulseShape.txt")
#process.EcalTrivialConditionRetriever.APDSimPulseShapeFile = cms.untracked.string("APD_SimPulseShape.txt")

### phase II Pulse Shapes
process.EcalTrivialConditionRetriever.sim_pulse_shape_TI = cms.untracked.double(0.250)
process.EcalTrivialConditionRetriever.sim_pulse_shape_EB_thresh = cms.double(0.201244)
process.EcalTrivialConditionRetriever.sim_pulse_shape_EE_thresh = cms.double(0.201244)
process.EcalTrivialConditionRetriever.sim_pulse_shape_APD_thresh = cms.double(0.201244)
process.EcalTrivialConditionRetriever.EBSimPulseShapeFile = cms.untracked.string("EB_SimPulseShape_PhaseII.txt")
process.EcalTrivialConditionRetriever.EESimPulseShapeFile = cms.untracked.string("EB_SimPulseShape_PhaseII.txt")
process.EcalTrivialConditionRetriever.APDSimPulseShapeFile = cms.untracked.string("EB_SimPulseShape_PhaseII.txt")


process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_31X_ECAL'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:DB.db'

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
      record = cms.string('EcalSimPulseShapeRcd'),
      tag = cms.string('EcalSimPulseShape_default_mc')
    )
  )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
  timetype = cms.string('runnumber'),
  toCopy = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalSimPulseShapeRcd'),
      container = cms.string('EcalSimPulseShape')
    )
  )
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)
