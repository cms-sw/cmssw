#
# cfg file to run on L1 GT output file, with GCT and GMT EDM products included
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestHLTLevel1GTSeed')

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(10)
)

process.source = cms.Source('PoolSource',
    fileNames=cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/test_HLTLevel1GTSeed_source.root')
)

#/RelValTTbar/CMSSW_3_1_0_pre3_IDEAL_30X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
process.PoolSource.fileNames = [
       '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/1EE2ECB6-170A-DE11-BE8C-0016177CA7A0.root',
       '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/2863BE9C-2A0A-DE11-9452-000423D996C8.root',
       '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/2AD4CDCD-170A-DE11-AF56-000423D9A2AE.root',
       '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/2EB5C3F1-2C0A-DE11-8ED9-000423D999CA.root'
]

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'IDEAL_30X::All'

#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_Test_cff')

#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_cff')
#
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v0_L1T_Scales_20080922_Imp0_Unprescaled_cff')

#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff')
#
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_v1_L1T_Scales_20080922_Imp0_Unprescaled_cff')

#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v4_Unprescaled_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v5_Unprescaled_cff')
#
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v1_Unprescaled_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff')
#
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v3_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
#process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v4_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')

# Global Trigger emulator to produce the trigger object maps

import L1Trigger.GlobalTrigger.gtDigis_cfi
process.hltL1GtObjectMap = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()


# input tag for GMT readout collection: 
#     gmtDigis = GMT emulator (default)
#     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
process.hltL1GtObjectMap.GmtInputTag = 'hltGtDigis'

# input tag for GCT readout collections: 
#     gctDigis = GCT emulator (default) 
process.hltL1GtObjectMap.GctInputTag = 'hltGctDigis'

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
process.l1extraParticles.muonSource = cms.InputTag('hltGtDigis')
process.l1extraParticles.isolatedEmSource = cms.InputTag('hltGctDigis', 'isoEm')
process.l1extraParticles.nonIsolatedEmSource = cms.InputTag('hltGctDigis', 'nonIsoEm')
process.l1extraParticles.centralJetSource = cms.InputTag('hltGctDigis', 'cenJets')
process.l1extraParticles.forwardJetSource = cms.InputTag('hltGctDigis', 'forJets')
process.l1extraParticles.tauJetSource = cms.InputTag('hltGctDigis', 'tauJets')
process.l1extraParticles.etTotalSource = cms.InputTag('hltGctDigis')
process.l1extraParticles.etHadSource = cms.InputTag('hltGctDigis')
process.l1extraParticles.etMissSource = cms.InputTag('hltGctDigis')
process.l1extraParticles.htMissSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.hfRingEtSumsSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.hfRingBitCountsSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.ignoreHtMiss = cms.bool(True)

# this module
process.load('HLTrigger.HLTfilters.hltLevel1GTSeed_cfi')
 
# replacing arguments for hltLevel1GTSeed

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

logExpressionNumber = 12

if logExpressionNumber == 0 :
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet50'                                     # 0
elif logExpressionNumber == 1 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet50U'                                    # 1
elif logExpressionNumber == 2 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_HTT250 OR L1_ETM10 OR L1_ETT60 OR L1_SingleEG15' # 2
elif logExpressionNumber == 3 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_HTT250 OR L1_HTT300 OR L1_SingleEG15'            # 3
elif logExpressionNumber == 4 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'NOT L1_SingleEG15'                                  # 4
elif logExpressionNumber == 5 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_ZeroBias'                                        # 5
elif logExpressionNumber == 6 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleEG15 AND L1_HTT300'                        # 6
elif logExpressionNumber == 7 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleEG15 AND (L1_HTT300 AND NOT L1_SingleMu7)' # 7
elif logExpressionNumber == 8 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '(L1_SingleEG15 OR L1_QuadJet40) AND (L1_HTT300 AND NOT L1_SingleMu7)'  # 8
elif logExpressionNumber == 9 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '(L1_SingleEG15 OR L1_QuadJet40) AND ((L1_HTT300 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)' # 9
elif logExpressionNumber == 10 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet6U'                                     # 10
elif logExpressionNumber == 11 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet6U'                                     # 11
elif logExpressionNumber == 12 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_IsoEG10_Jet6U_ForJet6U'                          # 12
elif logExpressionNumber == 13 :        
    # for technical triggers, one specifies by bit number        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '1 AND 15 AND NOT (29 OR 55)'                        # 13
elif logExpressionNumber == 14 :
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '(L1_SingleEG15 OR L1_QuadJet6U) AND ((L1_HTT200 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)' # 14
else :
    print 'Error: no logical expression defined'    

    
# InputTag for the L1 Global Trigger DAQ readout record
#   GT Emulator = gtDigis
#   GT Unpacker = l1GtUnpack
#
#   cloned GT unpacker in HLT = gtDigis
#
process.hltLevel1GTSeed.L1GtReadoutRecordTag = cms.InputTag('hltGtDigis')
    
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
#process.hltLevel1GTSeed.saveTags = cms.untracked.bool(True)


# path to be run
process.p = cms.Path(process.hltL1GtObjectMap * process.L1Extra * process.hltLevel1GTSeed)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['hltLevel1GTSeed']
process.MessageLogger.categories = ['HLTLevel1GTSeed']
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.untracked.PSet(
    threshold=cms.untracked.string('DEBUG'),
    #threshold = cms.untracked.string('INFO'),
    #threshold = cms.untracked.string('ERROR'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)     ## DEBUG, all messages
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(2)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    )
)


