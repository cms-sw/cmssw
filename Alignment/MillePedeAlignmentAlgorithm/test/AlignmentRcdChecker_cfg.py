import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("scan")
options = VarParsing.VarParsing("analysis")
options.register ('inputSqliteFile',
                  "alignments_MP.db",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "input sql file")

options.parseArguments()

process.load("CondCore.CondDB.CondDB_cfi")

### new alignment to check
process.CondDB.connect = 'sqlite_file:'+options.inputSqliteFile
process.newTrackerAlignment = cms.ESSource("PoolDBESSource",process.CondDB,
                                           toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                      tag = cms.string("Alignments"),
                                                                      label = cms.untracked.string("toCheck"))))

process.es_prefer_newTrackerAlignment = cms.ESPrefer("PoolDBESSource","newTrackerAlignment")

### reference alignment
process.CondDB.connect = 'frontier://FrontierPrep/CMS_CONDITIONS'
process.refTrackerAlignment = cms.ESSource("PoolDBESSource",process.CondDB,
                                           toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                      tag = cms.string("TrackerAlignmentForUnitTestChecks_v0"))))
process.es_prefer_refTrackerAlignment = cms.ESPrefer("PoolDBESSource","refTrackerAlignment")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1), # do not change!
                            firstRun = cms.untracked.uint32(1))

process.AlignmentRcdScan = cms.EDAnalyzer("AlignmentRcdChecker")
process.AlignmentRcdScan.verbose = cms.bool(True) 
process.AlignmentRcdScan.label = cms.string("toCheck") 
process.AlignmentRcdScan.compareStrict = cms.bool(True) 

process.p = cms.Path(process.AlignmentRcdScan)
