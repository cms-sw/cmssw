#
# cfg file to run on L1 GT output file, with GCT and GMT EDM products included
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestHLTLevel1GTSeed')

useMC = False

# explicit choice of the L1 menu,
# use l1Menu='' for default menu from GT
l1Menu = ''

# other available choices (must be compatible with the GlobalTag)
#l1Menu = 'L1Menu_Commissioning2009_v0'
#l1Menu = 'L1Menu_MC2009_v0'
#l1Menu = 'L1Menu_startup2_v4'
#l1Menu = 'L1Menu_2008MC_2E30'

# private menu (must edit the corresponding part in the menu list)
# must be compatible with the GlobalTag
#l1Menu = 'myMenu'

###################### end user choices ###################

# number of events to be processed and source file
process.maxEvents.input = 100

process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())

if useMC:
    gtName = 'auto:run2_mc_l1stage1'
    process.source.fileNames = ['/store/relval/CMSSW_8_0_0/RelValTTbar_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v4-v1/10000/42D6DF66-9DDA-E511-9200-0CC47A4D7670.root']
else:
    gtName = 'auto:run2_data'
    process.source.fileNames = ['/store/data/Run2015D/SingleMuon/RAW-RECO/ZMu-PromptReco-v3/000/256/677/00000/0EC78D60-1E5F-E511-B77F-02163E0123C5.root']

# load and configure modules via GlobalTag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, gtName, '')

# [UNTESTED] explicit choice of the L1 menu, overwriting the GlobalTag menu
if l1Menu != '':

    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')

    if l1Menu == 'L1Menu_MC2009_v2':
        process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v2_L1T_Scales_20080922_Imp0_Unprescaled_cff')

    elif l1Menu == 'L1Menu_2008MC_2E30':
        process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")

    elif l1Menu == 'L1Menu_Commissioning2009_v0':
        process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')

    elif l1Menu == 'myMenu':
        process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v4_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff")

    else :
        print('No such L1T menu: ', l1Menu)
        sys.exit(1)

else :
    print('# Using default L1 trigger menu from GlobalTag:', gtName)


# Global Trigger emulator to produce the trigger object maps

import L1Trigger.GlobalTrigger.gtDigis_cfi
process.hltL1GtObjectMap = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()

# input tag for GMT readout collection: 
#     gmtDigis = GMT emulator (default)
#     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
process.hltL1GtObjectMap.GmtInputTag = 'gtDigis'

# input tag for GCT readout collections: 
#     gctDigis = GCT emulator (default) 
process.hltL1GtObjectMap.GctInputTag = 'gctDigis'

# input tag for CASTOR record 
#     castorL1Digis =  CASTOR
#process.hltL1GtObjectMap.CastorInputTag = cms.InputTag("castorL1Digis")
    
# technical triggers: a vector of input tags, one tag per each technical 
# trigger producer 
# 
# by default: empty vector
    
# Example:
# TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('aTechTrigDigis'), 
#                                            cms.InputTag('anotherTechTriggerDigis')),
process.hltL1GtObjectMap.TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('bscTrigger'))

# logical flag to produce the L1 GT DAQ readout record
#     if true, produce the record (default)
process.hltL1GtObjectMap.ProduceL1GtDaqRecord = False
    
# logical flag to produce the L1 GT EVM readout record
#     if true, produce the record (default)
process.hltL1GtObjectMap.ProduceL1GtEvmRecord = False

# logical flag to produce the L1 GT object map record
#     if true, produce the record (default)
#process.hltL1GtObjectMap.ProduceL1GtObjectMapRecord = False

# logical flag to write the PSB content in the  L1 GT DAQ record
#     if true, write the PSB content in the record (default)
process.hltL1GtObjectMap.WritePsbL1GtDaqRecord = False

# logical flag to read the technical trigger records
#     if true, it will read via getMany the available records (default)
process.hltL1GtObjectMap.ReadTechnicalTriggerRecords = True

# number of "bunch crossing in the event" (BxInEvent) to be emulated
# symmetric around L1Accept (BxInEvent = 0):
#    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
# even numbers (except 0) "rounded" to the nearest lower odd number
# negative value: emulate TotalBxInEvent as given in EventSetup  
process.hltL1GtObjectMap.EmulateBxInEvent = 1

# length of BST record (in bytes) from parameter set
# negative value: take the value from EventSetup      
process.hltL1GtObjectMap.BstLengthBytes = -1

# L1 Extra
process.load('L1Trigger.Configuration.L1Extra_cff')

# replacing arguments for L1Extra
process.l1extraParticles.muonSource = cms.InputTag('gtDigis')
process.l1extraParticles.isolatedEmSource = cms.InputTag('gctDigis', 'isoEm')
process.l1extraParticles.nonIsolatedEmSource = cms.InputTag('gctDigis', 'nonIsoEm')
process.l1extraParticles.centralJetSource = cms.InputTag('gctDigis', 'cenJets')
process.l1extraParticles.forwardJetSource = cms.InputTag('gctDigis', 'forJets')
process.l1extraParticles.tauJetSource = cms.InputTag('gctDigis', 'tauJets')
process.l1extraParticles.etTotalSource = cms.InputTag('gctDigis')
process.l1extraParticles.etHadSource = cms.InputTag('gctDigis')
process.l1extraParticles.etMissSource = cms.InputTag('gctDigis')
process.l1extraParticles.htMissSource = cms.InputTag("gctDigis")
process.l1extraParticles.hfRingEtSumsSource = cms.InputTag("gctDigis")
process.l1extraParticles.hfRingBitCountsSource = cms.InputTag("gctDigis")
process.l1extraParticles.ignoreHtMiss = cms.bool(False)
process.l1extraParticles.centralBxOnly = cms.bool(False)

# this module
process.load('HLTrigger.HLTfilters.hltLevel1GTSeed_cfi')

# replacing arguments for hltLevel1GTSeed

# default: true
#    seeding done via L1 trigger object maps, with objects that fired 
#    only objects from the central BxInEvent (L1A) are used
# if false:
#    seeding is done ignoring if a L1 object fired or not, 
#    adding all L1EXtra objects corresponding to the object types 
#    used in all conditions from the algorithms in logical expression 
#    for a given number of BxInEvent
process.hltLevel1GTSeed.L1UseL1TriggerObjectMaps = cms.bool(False)
#
# option used forL1UseL1TriggerObjectMaps = False only
# number of BxInEvent: 1: L1A=0; 3: -1, L1A=0, 1; 5: -2, -1, L1A=0, 1, 2
process.hltLevel1GTSeed.L1NrBxInEvent = cms.int32(3)

# seeding done via technical trigger bits, if value is 'True';
# default: false (seeding via physics algorithms)
#process.hltLevel1GTSeed.L1TechTriggerSeeding = True

# seeding done with aliases for physics algorithms
#process.hltLevel1GTSeed.L1UseAliasesForSeeding = cms.bool(False)

# logical expression for the required L1 algorithms;
# the algorithms are specified by name
# allowed operators: 'AND', 'OR', 'NOT', '(', ')'
#
# by convention, 'L1GlobalDecision' logical expression means global decision
# 

l1SeedLogicalExprs = [
    'L1_SingleJet50',                                     # 0
    'L1_SingleJet50U',                                    # 1
    'L1_HTT200 OR L1_ETM20 OR L1_ETT60 OR L1_SingleEG15', # 2
    'L1_HTT250 OR L1_HTT300 OR L1_SingleEG15',            # 3
    'NOT L1_SingleEG15',                                  # 4
    'L1_ZeroBias',                                        # 5
    'L1_SingleEG15 AND L1_HTT200',                        # 6
    'L1_SingleEG15 AND (L1_HTT100 OR L1_SingleMu0) OR L1_SingleJet6U', # 7
    '(L1_SingleEG15 OR L1_QuadJet40) AND (L1_HTT300 AND NOT L1_SingleMu7)', # 8
    '(L1_SingleEG15 OR L1_QuadJet40) AND ((L1_HTT300 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)', # 9
    'L1_SingleJet6U',                                     # 10
    'L1_SingleJet6U',                                     # 11
    'L1_IsoEG10_Jet6U_ForJet6U',                          # 12
    '1 AND 15 AND NOT (29 OR 55)',                        # 13 (for technical triggers, one specifies by bit number)
    '(L1_SingleEG15 OR L1_QuadJet6U) AND ((L1_HTT200 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)', # 14
]

process.hltLevel1GTSeed.L1SeedsLogicalExpression = l1SeedLogicalExprs[6]

# InputTag for the L1 Global Trigger DAQ readout record
#   GT Emulator = gtDigis
#   GT Unpacker = l1GtUnpack
#
#   cloned GT unpacker in HLT = gtDigis
#
process.hltLevel1GTSeed.L1GtReadoutRecordTag = cms.InputTag('gtDigis')

# InputTag for L1 Global Trigger object maps
#   only the emulator produces the object maps
#   GT Emulator = gtDigis
#
#   cloned GT emulator in HLT = l1GtObjectMap
#
process.hltLevel1GTSeed.L1GtObjectMapTag = cms.InputTag('hltL1GtObjectMap')

# InputTag for L1 particle collections (except muon)
#   L1 Extra = l1extraParticles
#
#process.hltLevel1GTSeed.L1CollectionsTag = cms.InputTag('l1extraParticles')

# InputTag for L1 muon collection
#process.hltLevel1GTSeed.L1MuonCollectionTag = cms.InputTag('l1extraParticles')

# saveTags for AOD book-keeping
#process.hltLevel1GTSeed.saveTags = cms.bool( True )

# path to be run
process.p = cms.Path(process.hltL1GtObjectMap * process.L1Extra * process.hltLevel1GTSeed)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['hltLevel1GTSeed']
process.MessageLogger.HLTLevel1GTSeed = dict()

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.HLTLevel1GTSeed = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
