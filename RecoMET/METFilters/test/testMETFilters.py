##____________________________________________________________________________||
import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("FILT")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger")

##____________________________________________________________________________||
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.inputFiles = 'file:/shome/clange/ChargedHadronMuonRefFilter/CMSSW_7_4_15_patch1/src/RecoMET/METFilters/test/pickevents_v3RECO.root', 
options.outputFile = 'filtered.root'
options.maxEvents = -1
options.parseArguments()

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.options   = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

##____________________________________________________________________________||
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.GlobalTag.globaltag = cms.string("74X_dataRun2_Prompt_v3")

##____________________________________________________________________________||
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
    )

##____________________________________________________________________________||
process.load("RecoMET.METFilters.metFilters_cff")

process.p = cms.Path(
    process.metFilters
)

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string(options.outputFile),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
    )

process.outpath = cms.EndPath(process.out)

##____________________________________________________________________________||