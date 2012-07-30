import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.MessageLogger.cerr.FwkReport.reportEvery = 10000

#INPUTFILE="input.root"
INPUTFILE="rfio:/castor/cern.ch/user/j/jalimena/177139/HT/out_177139_HT_0.root"
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(INPUTFILE)
                            )

process.load("HLTrigger.HLTanalyzers.hltOfflineReproducibility_cfi")

OUTPUTFILE="./HLTOfflineReproducibility_177139HT_0.root"
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(OUTPUTFILE)
                                   )

process.p = cms.Path(process.hltOfflineReproducibility)
