import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")

process.task = cms.Task()

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## Options and output report
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source (expect a locally produced patTuple.root or miniAODTuple.root, produced via standard sequesnted in PatAlgos)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:patMiniAOD_standard.root")
)
## Maximal number of events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
## Load dump of tau variables
process.load("RecoTauTag.Configuration.dumpTauVariables_cfi")
process.task.add(process.dumpTauVariables)
## Define variables to be dumped to out
from RecoTauTag.Configuration.tauVariables_cff import slimmedVariables
process.dumpTauVariables.variables = slimmedVariables

## OutputModule configuration (expects a path 'p')
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('dumpTauVariables_slimmed.root'),
                               outputCommands = cms.untracked.vstring('drop *', 'keep *_*_*_DUMP' )
                               )
process.outpath = cms.EndPath(process.out, process.task)
