# cfg file to test L1 GT stable parameters

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('L1RCTGlobalTagTest')

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


process.load('L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff')

process.l1RCTParametersTest = cms.EDAnalyzer("L1RCTParametersTester")
process.l1RCTChannelMaskTest = cms.EDAnalyzer("L1RCTChannelMaskTester")

# paths to be run


process.p = cms.Path(process.l1RCTParametersTest
                     *process.l1RCTChannelMaskTest
)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['RCTConfig']
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
