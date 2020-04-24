import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
#         threshold = cms.untracked.string( 'ERROR' )
        threshold = cms.untracked.string( 'WARNING' )
    )
)
process.SimpleMemoryCheck = cms.Service( "SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32( 0 )
)

# Magnetic Field
process.load( "Configuration.StandardSequences.MagneticField_xMAG_FIELDx_cff" )

# Geometry
process.load( "Configuration.StandardSequences.GeometryRecoDB_cff" )

# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'xGLOBAL_TAGx'
process.es_prefer_GlobalTag = cms.ESPrefer(
    'PoolDBESSource',
    'GlobalTag'
)

# SiStrip DQM
process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

# Input
process.load( "xINCLUDE_DIRECTORYx.inputFiles_cff" )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

# HLT Filter
process.hltFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths           = cms.vstring(
        'HLT_WhatEverFunnyFilter',
        'HLT_TrackerCosmics',
        'HLT_TrackerCosmics_CoTF',
        'HLT_TrackerCosmics_RS'  ,
        'HLT_TrackerCosmics_CTF'
    ),
    eventSetupPathsKey = cms.string( '' ),
    andOr              = cms.bool( True ),
    throw              = cms.bool( False ),
    # use this according to https://hypernews.cern.ch/HyperNews/CMS/get/global-runs/537.html
    # TO BE TEMPLATYFIED!
    TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'HLT' )
#     TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'FU' )    
)

# Scheduling
process.p = cms.Path(
xHLT_FILTERxprocess.hltFilter                            *
xRECO_FROM_RAWxprocess.SiStripDQMRecoFromRaw                *
xDQM_FROM_RAWxprocess.SiStripDQMSourceGlobalRunCAF_fromRAW *
    process.SiStripDQMRecoGlobalRunCAF           *
#     process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripMonitorClusterCAF             *
    process.MEtoEDMConverter
)

# Output
process.out = cms.OutputModule( "PoolOutputModule",
    fileName       = cms.untracked.string( 'xOUTPUT_DIRECTORYx/SiStripDQMOfflineGlobalRunCAF-xRUN_NAMEx.root' ),
    SelectEvents   = cms.untracked.PSet(
        SelectEvents = cms.vstring( 'p' )
    ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_SiStripDQMOfflineGlobalRunCAF'
    )
)

process.outpath = cms.EndPath(
    process.out
)
