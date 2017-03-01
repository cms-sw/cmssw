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
        threshold = cms.untracked.string( 'ERROR' )
    )
)

### Import ###

# Magnetic fiels #
process.load( "Configuration.StandardSequences.MagneticField_xMAG_FIELDx_cff" )
# Geometry #
process.load( "Configuration.StandardSequences.GeometryRecoDB_cff" )
# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
# process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'xGLOBAL_TAGx'
process.es_prefer_GlobalTag = cms.ESPrefer(
    'PoolDBESSource',
    'GlobalTag'
)

### Input ###

# Source #
process.load( "xINCLUDE_DIRECTORYx.inputFilesCAF_cff" )
# Input steering #
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

### SiStrip DQM ###

process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

# DQM saver #
process.dqmSaver.dirName = 'xMERGE_PATHx'

### Scheduling ###

process.p = cms.Path(
    process.EDMtoMEConverter        *
    process.SiStripOfflineDQMClient *
    process.qTester                 *
    process.dqmSaver
)
