import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.MessageLogger.cerr.FwkReport.reportEvery = 10000

INPUTFILE="rfio:/castor/cern.ch/user/j/jalimena/177139/HT/out_177139_HT_0.root"
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(INPUTFILE)
                            )

process.load("HLTrigger.HLTanalyzers.hltOfflineReproducibility_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['hltonline']
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

OUTPUTFILE="./HLTOfflineReproducibility_177139HT_0.root"
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(OUTPUTFILE)
                                   )

process.p = cms.Path(process.hltOfflineReproducibility)
