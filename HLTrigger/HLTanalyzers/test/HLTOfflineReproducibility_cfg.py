import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.MessageLogger.cerr.FwkReport.reportEvery = 10000

INPUTFILE="input.root"
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(INPUTFILE)
                            )

process.load("HLTrigger.HLTanalyzers.HLTOfflineReproducibility_cfi")

OUTPUTFILE = 'reproducbility.out'
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(OUTPUTFILE)
                                   )

process.p = cms.Path(process.hltofflinereproducibility)
