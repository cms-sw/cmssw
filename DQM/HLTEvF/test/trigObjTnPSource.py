import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

# DQM service
process.load("DQMServices.Core.DQMStore_cfi")

# MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
#process.MessageLogger.cerr.INFO.limit = 1000
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis') 
options.register('dqmTag','/HLT/TrigObjTnpSource/All',options.multiplicity.singleton,options.varType.string," whether we are running on miniAOD or not")
options.parseArguments()


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True ),
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 4 ),
)

# Source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(options.inputFiles)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.load("DQM.HLTEvF.trigObjTnPSource_cfi")

process.load('DQMServices.Components.DQMFileSaver_cfi')
process.dqmSaver.workflow = options.dqmTag

process.endp = cms.EndPath( process.trigObjTnPSource + process.dqmSaver )
