import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'ERROR' )
    )
)
process.SimpleMemoryCheck = cms.Service( "SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32( 0 )
)

# Magnetic Field
process.load( "Configuration.StandardSequences.MagneticField_xMAG_FIELDx_cff" )

# Geometry
process.load( "Configuration.StandardSequences.Geometry_cff" )

# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.connect   = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "xGLOBAL_TAGx"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

# SiStrip DQM
process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

# Input
process.load( "xINCLUDE_DIRECTORYx.inputFiles_cff" )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

# HLT Filter
process.hltFilter = cms.EDFilter( "HLTHighLevel",
    HLTPaths          = cms.vstring(
        'HLT_TrackerCosmics_CoTF',
        'HLT_TrackerCosmics_CTF' ,
        'HLT_TrackerCosmics_RS'
    ),
    andOr             = cms.bool( True ),
    TriggerResultsTag = cms.InputTag( "TriggerResults", "", "FU" )
)

# Output
process.out = cms.OutputModule( "PoolOutputModule",
    fileName       = cms.untracked.string( 'xOUTPUT_DIRECTORYx/SiStripDQMOfflineGlobalRunCAF-xRUN_NAMEx.root' ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_SiStripDQMOfflineGlobalRunCAF'
    )
)

# Scheduling
process.p = cms.Path(
xRECO_FROM_RAWxprocess.SiStripDQMRecoFromRaw                *
xHLT_FILTERxprocess.hltFilter                            *
xDQM_FROM_RAWxprocess.SiStripDQMSourceGlobalRunCAF_fromRAW *
    process.SiStripDQMRecoGlobalRunCAF           *
    process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.MEtoEDMConverter
)

process.outpath = cms.EndPath(
    process.out
)
