import FWCore.ParameterSet.Config as cms

process = cms.Process("scan")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000000)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200000)
)
process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1), # do not change!
                            firstRun = cms.untracked.uint32(1)
                            )

process.GlobalTag.globaltag = 'GR_R_52_V7::All'

process.AlignmentRcdScan = cms.EDAnalyzer("AlignmentRcdScan")
process.AlignmentRcdScan.verbose = cms.untracked.bool(False) #True) 

process.AlignmentRcdScan.mode = cms.untracked.string('Tk')
#process.AlignmentRcdScan.mode = cms.untracked.string('DT')
#process.AlignmentRcdScan.mode = cms.untracked.string('CSC')

process.p = cms.Path(process.AlignmentRcdScan)

