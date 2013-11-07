import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## global tag
process.GlobalTag.globaltag = 'START61_V8::All'
##process.GlobalTag.globaltag = 'GR_R_38X_V13::All' 
#process.GlobalTag.globaltag   = 'GR10_P_V10::All'
#process.GlobalTag.globaltag   = 'GR_R_52_V7::All'

## input file (adapt input file name correspondingly)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:topDQM_production.root"),
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
process.dqmSaver.workflow       = cms.untracked.string('/TopVal/CMSSW_3_8_4/RECO') ## adapt apropriately


## path definitions
process.edmtome = cms.Path(
    process.EDMtoME
)
process.dqmsave = cms.Path(
    process.DQMSaver
)

## schedule definition
process.schedule = cms.Schedule(process.edmtome,process.dqmsave)
