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
process.GlobalTag.globaltag = 'CRAFT_V4P::All'
process.es_prefer_GlobalTag = cms.ESPrefer( 'PoolDBESSource', 'GlobalTag' )

### SiStrip DQM ###

process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

### Input ###

# Source #
process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        # run 66615, prompt reconstruction
        'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/v1/000/066/615/60006306-D99B-DD11-AE94-001617E30F58.root',
        'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/v1/000/066/615/42770840-E79B-DD11-AB9B-0016177CA7A0.root',
        'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/v1/000/066/615/AA2F3B3D-E79B-DD11-B9D4-001617E30D12.root',
        'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/v1/000/066/615/D02321A7-D79B-DD11-917C-001617C3B76A.root',
        'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/v1/000/066/615/54529935-E79B-DD11-86A4-001617DBD316.root'
    )
)
# Input steering #
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 25000 )
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
    fileName       = cms.untracked.string( './SiStripDQMOfflineGlobalRunCAF.root' ),
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
