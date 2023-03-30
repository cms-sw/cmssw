import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


## VarParsing object

options = VarParsing('python')

options.register('globaltag', '121X_mcRun3_2021_realistic_v15', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Set Global Tag')
options.register('name', 'TEST', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Set process name')

options.parseArguments()


## Process

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## global tag
process.GlobalTag.globaltag = cms.string(options.globaltag)

## input file 
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:EXOTICA_DQM_TEST.root'),
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
#process.DQMStore.collateHistograms        = True
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun  = True
process.dqmSaver.saveByRun      = -1
process.dqmSaver.saveAtJobEnd   = True
process.dqmSaver.forceRunNumber = 1
process.dqmSaver.workflow       = '/Physics/Exotica/' + options.name ## adapt appropriately


## path definitions
process.edmtome = cms.Path(
    process.EDMtoME
)
process.dqmsave = cms.Path(
    process.DQMSaver
)

## schedule definition
process.schedule = cms.Schedule(process.edmtome,process.dqmsave)
