import FWCore.ParameterSet.Config as cms

process = cms.Process("COC")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:patTuple.root")
)

## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
                                
## let it run
process.load("PhysicsTools.PatExamples.customizedSelection_cff")
process.load("PhysicsTools.PatExamples.customizedCOC_cff")

process.p = cms.Path(
    process.customSelection
   *process.customCOC
    )

from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('cocTuple.root'),
                               # save only events passing the full path
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               # save PAT Layer 1 output; you need a '*' to
                               # unpack the list of commands 'patEventContent'
                               outputCommands = cms.untracked.vstring(
                                   'keep *',
                                  #'drop *_selectedPatElectrons_*_*',
                                   'drop *_selectedPatJets_*_*',
                                   'keep *_*_caloTowers_*',
                                   'keep *_*_genJets_*')
                               )

process.outpath = cms.EndPath(process.out)
