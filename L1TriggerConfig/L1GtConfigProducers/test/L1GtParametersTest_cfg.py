# cfg file to test L1 GT parameters

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtParametersTest")
process.l1GtParametersTest = cms.EDAnalyzer("L1GtParametersTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtParametersConfig_cff")


# path to be run
process.p = cms.Path(process.l1GtParametersTest)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtParametersTest']
process.MessageLogger.cout = cms.untracked.PSet(
    threshold=cms.untracked.string('DEBUG'),
    #threshold = cms.untracked.string('INFO'),
    #threshold = cms.untracked.string('ERROR'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    default = cms.untracked.PSet( 
        limit=cms.untracked.int32(-1)  
    )
)
