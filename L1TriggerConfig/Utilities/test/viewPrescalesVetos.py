from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('db',
                 'static',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Source DB: prod/prep/static/sqlite"
)
options.register('run',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run (IOV)"
)
options.parseArguments()

if options.db == "static" :
    process.load('L1Trigger.L1TGlobal.PrescalesVetos_cff')
    #process.L1TGlobalPrescalesVetos.PrescaleXMLFile   = cms.string('UGT_BASE_RS_PRESCALES.xml')
    #process.L1TGlobalPrescalesVetos.AlgoBxMaskXMLFile = cms.string('UGT_BASE_RS_ALGOBX_MASK.xml')
    #process.L1TGlobalPrescalesVetos.FinOrMaskXMLFile  = cms.string('UGT_BASE_RS_FINOR_MASK.xml')
    #process.L1TGlobalPrescalesVetos.VetoMaskXMLFile   = cms.string('UGT_BASE_RS_VETO_MASK.xml')
else :
    if   options.db == "prod" :
        sourceDB = "frontier://FrontierProd/CMS_CONDITIONS"
    elif options.db == "prep" :
        sourceDB = "frontier://FrontierPrep/CMS_CONDITIONS"
    elif "sqlite" in options.db :
        sourceDB = options.db
    else :
        print("Unknown input DB: ", options.db, " should be static/prod/prep/sqlite:...")
        exit(0)

    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = cms.string(sourceDB)
    process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TGlobalPrescalesVetosRcd'),
                 tag = cms.string("L1TGlobalPrescalesVetos_passThrough_mc")
#                 tag = cms.string("L1TGlobalPrescalesVetos_Stage2v0_hlt")
            )
       )
    )

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(options.run))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.l1gpv = cms.EDAnalyzer("L1TGlobalPrescalesVetosViewer",
    prescale_table_verbosity = cms.untracked.int32(0),
    bxmask_map_verbosity     = cms.untracked.int32(0),
    veto_verbosity           = cms.untracked.int32(0)
)

process.p = cms.Path(process.l1gpv)

