import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')

## input file (adapt input file name correspondingly)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:b2gDQM.root"),
    processingMode = cms.untracked.string('RunsAndLumis')
)

## number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

## output options
process.options = cms.untracked.PSet(
    Rethrow  = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

## DQMStore and output configuration
process.DQMStore.collateHistograms        = True
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun  = True
process.dqmSaver.saveByRun      = cms.untracked.int32( -1)
process.dqmSaver.saveAtJobEnd   = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(  1)
process.dqmSaver.workflow       = cms.untracked.string('/B2G/CMSSW_7_2_0_pre5/B2GDQM') ## adapt apropriately


## path definitions
process.edmtome = cms.Path(
    process.EDMtoME
)
process.dqmsave = cms.Path(
    process.DQMSaver
)

## schedule definition
process.schedule = cms.Schedule(process.edmtome,process.dqmsave)
