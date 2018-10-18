from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('db',
                 'static:caloStage2Params_2017_v1_4_inconsistent_cfi.py',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Source DB: prod/prep/static:.../sqlite:..."
)
options.register('run',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run (IOV)"
)
options.register('tag',
                 'L1TCaloParams_Stage2v3_hlt',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Run (IOV)"
)

options.parseArguments()

if "static" in options.db :
    process.load("L1Trigger.L1TCalorimeter." + options.db[7:])
else :
    if   options.db == "prod" :
        sourceDB = "frontier://FrontierProd/CMS_CONDITIONS"
    elif options.db == "prep" :
        sourceDB = "frontier://FrontierPrep/CMS_CONDITIONS"
    elif "sqlite" in options.db :
        sourceDB = options.db
    else :
        print("Unknown input DB: ", options.db, " should be static:.../prod/prep/sqlite:...")
        exit(0)

    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = cms.string(sourceDB)
    process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TCaloParamsRcd'),
                 tag = cms.string(options.tag)
            )
       )
   )

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(options.run))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.l1cpv = cms.EDAnalyzer("L1TCaloParamsViewer",
                               useStage2Rcd = cms.untracked.bool(False),
                               printPUSParams = cms.untracked.bool(False),
                               printTauCalibLUT = cms.untracked.bool(False),
                               printTauCompressLUT = cms.untracked.bool(False),
                               printJetCalibLUT = cms.untracked.bool(False),
                               printJetCalibPar = cms.untracked.bool(False),
                               printJetPUSPar = cms.untracked.bool(False),
                               printJetCompressPtLUT = cms.untracked.bool(False),
                               printJetCompressEtaLUT = cms.untracked.bool(False),
                               printEgCalibLUT = cms.untracked.bool(False),
                               printEgIsoLUT = cms.untracked.bool(False),
                               printEtSumMetPUSLUT = cms.untracked.bool(False),
                               printHfSF = cms.untracked.bool(False),
                               printHcalSF = cms.untracked.bool(False),
                               printEcalSF = cms.untracked.bool(False),
                               printEtSumEttPUSLUT = cms.untracked.bool(False),
                               printEtSumEcalSumPUSLUT = cms.untracked.bool(False),
                               printMetCalibrationLUT = cms.untracked.bool(False),
                               printMetHFCalibrationLUT = cms.untracked.bool(False),
                               printEtSumEttCalibrationLUT = cms.untracked.bool(False),
                               printEtSumEcalSumCalibrationLUT = cms.untracked.bool(False)
)

process.p = cms.Path(process.l1cpv)

