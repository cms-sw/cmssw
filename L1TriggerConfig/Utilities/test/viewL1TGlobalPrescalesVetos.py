import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(3))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.load('L1Trigger.L1TGlobal.PrescalesVetos_cff')
#process.L1TGlobalPrescalesVetos.PrescaleXMLFile   = cms.string('UGT_BASE_RS_PRESCALES.xml')
#process.L1TGlobalPrescalesVetos.AlgoBxMaskXMLFile = cms.string('UGT_BASE_RS_ALGOBX_MASK.xml')
#process.L1TGlobalPrescalesVetos.FinOrMaskXMLFile  = cms.string('UGT_BASE_RS_FINOR_MASK.xml')
#process.L1TGlobalPrescalesVetos.VetoMaskXMLFile   = cms.string('UGT_BASE_RS_VETO_MASK.xml')

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDBSetup,
       connect = cms.string('sqlite:./o2o/l1config.db'),
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TGlobalPrescalesVetosRcd'),
                 tag = cms.string("L1TGlobalPrescalesVetos_Stage2v0_hlt")
            )
       )
)

process.l1gpv = cms.EDAnalyzer("L1TGlobalPrescalesVetosViewer",
    prescale_table_verbosity = cms.untracked.int32(1),
    bxmask_map_verbosity     = cms.untracked.int32(1)
)

process.p = cms.Path(process.l1gpv)

