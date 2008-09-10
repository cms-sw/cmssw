import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'INFO' )
    )
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

# Magnetic Field
process.load( "Configuration.StandardSequences.MagneticField_0T_cff" )

# Geometry
process.load( "Configuration.StandardSequences.Geometry_cff" )

# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.connect   = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V5P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

# SiStrip DQM
process.load( "DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff" )

# HLT Filter
process.hltFilter = cms.EDFilter( "HLTHighLevel",
    HLTPaths          = cms.vstring(
        'CandHLTTrackerCosmicsCoTF',
        'CandHLTTrackerCosmicsRS'  ,
        'CandHLTTrackerCosmicsCTF'
    ),
    andOr             = cms.bool( True ),
    TriggerResultsTag = cms.InputTag( "TriggerResults", "", "FU" )
)

# Scheduling
process.p = cms.Path(
#     process.SiStripDQMRecoFromRaw                * # comment this out when running from RECO or with full reconstruction
#     process.hltFilter                            * # comment this out to switch off the HLT pre-selection
#     process.SiStripDQMSourceGlobalRunCAF_fromRAW * # comment this out when running from RECO or with full reconstruction
    process.SiStripDQMRecoGlobalRunCAF           *
    process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripDQMClientGlobalRunCAF         *
    process.qTester                              *
    process.dqmSaver
)

# Input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#       # RAW from CRUZET4 run 58733
#       '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/733/08D4E065-5E72-DD11-BB0A-0019B9F7310E.root' # 21279 events
      # RECO from CRUZET4 run 58733
      '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/058/733/127A54D3-6D72-DD11-84DF-000423D951D4.root' # 21221 events
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000 )
)
