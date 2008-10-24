import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

### Miscellanous ###

# Logging #
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool( True )
)
process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'INFO' )
    )
)

# # Profiling #
# process.ProfilerService = cms.Service( "ProfilerService",
#     paths = cms.untracked.vstring(
#         'FullEvent'
#     )
# )

# Memory check #
process.SimpleMemoryCheck = cms.Service( "SimpleMemoryCheck",
#     oncePerEventMode = cms.untracked.bool( True ),
    ignoreTotal      = cms.untracked.int32( 0 )
)

### Import ###

# Magnetic fiels #
process.load( "Configuration.StandardSequences.MagneticField_0T_cff" )
# Geometry #
process.load( "Configuration.StandardSequences.Geometry_cff" )
# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.connect   = 'frontier://PromptProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'CRAFT_V2P::All'
process.es_prefer_GlobalTag = cms.ESPrefer( 'PoolDBESSource', 'GlobalTag' )

### SiStrip DQM ###

process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

### Input ###

# Source #
process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        # run 62815, prompt reconstruction
#         'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/815/DC2578D7-D783-DD11-952C-000423D6C8E6.root', # 22509 events
#         'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/815/7A768201-D383-DD11-BABC-001617C3B6DE.root'  # 37902 events
        'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/815/5448D972-D783-DD11-BA04-000423D9997E.root',
        'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/815/5488DB89-DD83-DD11-A492-000423D99F3E.root'
    ),    
#     skipEvents = cms.untracked.uint32( 22000 )
    skipEvents = cms.untracked.uint32( 28000 )
)
# Input steering #
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000 )
)

# HLT Filter #
process.hltFilter = cms.EDFilter( "HLTHighLevel",
    HLTPaths          = cms.vstring(
        'HLT_TrackerCosmics_CoTF',
        'HLT_TrackerCosmics_CTF' ,
        'HLT_TrackerCosmics_RS'
    ),
    andOr             = cms.bool( True ),
    TriggerResultsTag = cms.InputTag( 'TriggerResults', '', 'FU' )
)

### Output ###

# DQM Saver path
process.dqmSaver.dirName = '.'

# PoolOutput #
process.out = cms.OutputModule( "PoolOutputModule",
#     fileName       = cms.untracked.string( '/afs/cern.ch/user/v/vadler/scratch0/cms/SiStripDQM/CMSSW_2_1_10/output/SiStripDQMOfflineGlobalRunCAF.root' ),
    fileName       = cms.untracked.string( './SiStripDQMOfflineGlobalRunCAF_test.root' ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_SiStripDQMOfflineGlobalRunCAF'
    )
)

### Scheduling ###

process.p = cms.Path(
#     process.SiStripDQMRecoFromRaw                * # comment this out when running from RECO or with full reconstruction
#     process.hltFilter                            * # comment this out to switch off the HLT pre-selection
#     process.SiStripDQMSourceGlobalRunCAF_fromRAW * # comment this out when running from RECO or with full reconstruction
    process.SiStripDQMRecoGlobalRunCAF           *
#     process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripMonitorClusterCAF             *
#     process.SiStripOfflineDQMClient              *
#     process.qTester                              *
#     process.dqmSaver
    process.MEtoEDMConverter
)

process.outpath = cms.EndPath(
    process.out
)
