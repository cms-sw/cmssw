import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMConvertOfflineGlobalRun" )

### Miscellanous ###

process.options = cms.untracked.PSet(
   fileMode    = cms.untracked.string( 'FULLMERGE' ),
   wantSummary = cms.untracked.bool( True )
)

# Logging #
process.MessageLogger = cms.Service( "MessageLogger",
    destination = cms.untracked.vstring(
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
process.load( "Configuration.StandardSequences.MagneticField_38T_cff" )
# Geometry #
process.load( "Configuration.StandardSequences.GeometryRecoDB_cff" )
# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
# process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'CRAFT_30X::All'
process.es_prefer_GlobalTag = cms.ESPrefer(
    'PoolDBESSource',
    'GlobalTag'
)

### Input ###

# Source #
process.source = cms.Source( "PoolSource",
    processingMode = cms.untracked.string( 'Runs' ),
    fileNames      = cms.untracked.vstring(
        'file1.root',
        'file2.root'
    )
)
# Input steering #
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

### SiStrip DQM ###

process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

### Output ###

# DQM store #
# process.DQMStore.referenceFileName = ''
# process.DQMStore.collateHistograms = False
# process.DQMStore.verbose           = 1

# EDM2ME #
# process.EDMtoMEConverter.convertOnEndLumi = False
# process.EDMtoMEConverter.convertOnEndRun  = True

# DQM saver #
process.dqmSaver.dirName = '.'

### Scheduling ###

process.p = cms.Path(
    process.EDMtoMEConverter        *
    process.SiStripOfflineDQMClient *
    process.qTester                 *
    process.dqmSaver
)
