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

# Magnetic Field
process.load( "Configuration.GlobalRuns.ForceZeroTeslaField_cff" )
# process.localUniform = cms.ESProducer( "UniformMagneticFieldESProducer",
#     ZFieldInTesla = cms.double( 0.0 )
# )
# process.prefer( "UniformMagneticFieldESProducer" )

# Geometry
process.load( "Configuration.StandardSequences.Geometry_cff" )

# Calibration 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.connect   = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
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
    process.SiStripDQMRecoGlobalRunCAF           *
    process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripDQMClientGlobalRunCAF         *
    process.qTester                              *
    process.dqmSaver
)

# Input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#       # RAW from MW33 run 56493
#       '/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/493/0053249A-456A-DD11-83FE-001D09F24448.root' # 25730 events
      # RECO from MW33 run 56493
      '/store/data/Commissioning08/Cosmics/RECO/MW33_v1/000/056/493/02DAAD57-616A-DD11-BBAB-001617E30E28.root' # 25773 events
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000 )
)
