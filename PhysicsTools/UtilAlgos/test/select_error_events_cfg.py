import FWCore.ParameterSet.Config as cms

process = cms.Process("ERRORSKIM")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarGENSIMRECO
process.source = cms.Source("PoolSource",
  fileNames = filesRelValTTbarGENSIMRECO,
  inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*") # drop the DQM histograms
)
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# create filter path on error messages
process.load("FWCore.Modules.logErrorFilter_cfi")
process.p = cms.Path( process.logErrorFilter )

## Output Module Configuration (expects a path 'p')
process.out = cms.OutputModule("PoolOutputModule",
                                fileName = cms.untracked.string('errorskim.root'),
                                # save only events passing the full path
                                SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                                outputCommands = cms.untracked.vstring("keep *")
                               )

process.outpath = cms.EndPath(process.out)

