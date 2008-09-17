#
# cfg file to run on L1 GT output file, with GCT and GMT EDM products included
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestHLTLevel1GTSeed')

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/test_HLTLevel1GTSeed_source.root')
)

#   /RelValQCD_Pt_80_120/CMSSW_2_1_8_STARTUP_V7_v1/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO
process.PoolSource.fileNames = [
       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/06C16DFA-9182-DD11-A4CC-000423D6CA6E.root',
       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/0A0241FB-9182-DD11-98E1-001617E30D40.root',
       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/12B41BA7-9282-DD11-9E7F-000423D94E70.root',
       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/14AD9BA0-9182-DD11-82D3-000423D987E0.root',
       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/1A5FD19D-9282-DD11-B0BB-000423D9970C.root'
]

# load and configure modules

# L1 EventSetup    
# select the appropriate configuration
#
# dummy configuration - trivial producers
process.load('L1Trigger.Configuration.L1DummyConfig_cff')
#
# DB configuration
#process.load('L1Trigger.Configuration.L1DBConfig_cff')

# WARNING: use always the same prescale factors and trigger mask for this module
#          and the modules used to obtain the input file 
#          (which is a L1 GMT + GCT + GT output file)!

# prescaled menu    
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_cff')

# uprescaled menu - change prescale factors to 1
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_Unprescaled_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_Unprescaled_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_Unprescaled_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1MenuTestCondCorrelation_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff')

# L1 Extra
process.load('L1Trigger.Configuration.L1Extra_cff')

# replacing arguments for L1Extra
process.l1extraParticles.muonSource = cms.InputTag("gtDigis")

# this module
process.load('HLTrigger.HLTfilters.hltLevel1GTSeed_cfi')
 
# replacing arguments for hltLevel1GTSeed

# seeding done via technical trigger bits, if value is 'True';
# default: false (seeding via physics algorithms)
#process.hltLevel1GTSeed.L1TechTriggerSeeding = True

# logical expression for the required L1 algorithms;
# the algorithms are specified by name
# allowed operators: 'AND', 'OR', 'NOT', '(', ')'
# 
process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
    'L1_SingleJet15'
    #'L1_HTT250 OR L1_HTT300 OR L1_SingleEG15'
    #'NOT L1_SingleEG15'
    #'L1_ZeroBias'
    #'L1_SingleEG15 AND L1_HTT300'
    #'L1_SingleEG15 AND (L1_HTT300 AND NOT L1_SingleMu7)'
    #'(L1_SingleEG15 OR L1_QuadJet40) AND (L1_HTT300 AND NOT L1_SingleMu7)'
    #'(L1_SingleEG15 OR L1_QuadJet40) AND ((L1_HTT300 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)'
        
# for technical triggers, one specifies by bit number        
#process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
#    '1 AND 15 AND NOT (29 OR 55)'

# InputTag for the L1 Global Trigger DAQ readout record
#   GT Emulator = gtDigis (default)
#   GT Unpacker = l1GtUnpack
#
#process.hltLevel1GTSeed.L1GtReadoutRecordTag = 'l1GtEmulDigis'

# InputTag for L1 Global Trigger object maps
#   only the emulator produces the object maps
#   GT Emulator = gtDigis
#   cloned GT emulator in HLT = l1GtObjectMap (default)
process.hltLevel1GTSeed.L1GtObjectMapTag = 'hltL1GtObjectMap'

# InputTag for L1 particle collections
#   L1 Extra = l1extraParticles (default)
#
#process.hltLevel1GTSeed.L1CollectionsTag = 'l1extraParticles'

# InputTag for L1 muon collection
#   L1 Extra = l1extraParticles (default)
#   Fast simulation = l1ParamMuons
#process.hltLevel1GTSeed.L1MuonCollectionTag = 'l1ParamMuons'

# path to be run
process.p = cms.Path(process.L1Extra*process.hltLevel1GTSeed)

# services

# Message Logger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = ['hltLevel1GTSeed']
process.MessageLogger.categories = ['*']
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.untracked.PSet(
    #threshold = cms.untracked.string('ERROR'),
    #threshold = cms.untracked.string('INFO'),
    #INFO = cms.untracked.PSet(
    #    limit = cms.untracked.int32(-1)
    #)#,
    threshold = cms.untracked.string('DEBUG'),
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)     ## DEBUG, all messages
    )
)

