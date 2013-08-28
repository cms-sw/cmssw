# cfg file to test L1 GT stable parameters

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('L1GtGlobalTagTest')

###################### user choices ######################

# choose the global tag type for RelVal; 
#     actual GlobalTag must be replaced in the 'if' below 
useGlobalTag = 'IDEAL_30X'
#useGlobalTag='STARTUP_30X'

###################### end user choices ###################

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source('EmptySource')

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
 
process.GlobalTag.globaltag = useGlobalTag + '::All'

# take the L1GTParameters from trivial producer instead of default Global Tag
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtParametersConfig_cff')
#process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtParametersTrivialProducer','l1GtParameters')

# paths to be run
process.l1GtStableParametersTest = cms.EDAnalyzer("L1GtStableParametersTester")
process.l1GtParametersTest = cms.EDAnalyzer('L1GtParametersTester')
process.l1GtBoardMapsTest = cms.EDAnalyzer("L1GtBoardMapsTester")
process.l1GtPsbSetupTest = cms.EDAnalyzer("L1GtPsbSetupTester")
process.l1GtPrescaleFactorsAndMasksTest = cms.EDAnalyzer("L1GtPrescaleFactorsAndMasksTester")
process.l1GtTriggerMenuTest = cms.EDAnalyzer("L1GtTriggerMenuTester")

process.p = cms.Path(process.l1GtStableParametersTest
                     *process.l1GtParametersTest
                     *process.l1GtBoardMapsTest
                     *process.l1GtPsbSetupTest
                     *process.l1GtPrescaleFactorsAndMasksTest
                     *process.l1GtTriggerMenuTest
)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['*']
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
