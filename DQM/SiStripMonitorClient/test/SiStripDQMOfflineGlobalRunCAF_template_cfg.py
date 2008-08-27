import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMOfflineGlobalRunCAF" )

process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'WARNING' )
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
process.load( "INCLUDE_DIRECTORY.SiStripDQMOfflineGlobalRunCAF_cff" )

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
RECO_FROM_RAW
HLT_FILTER
DQM_FROM_RAW
    process.SiStripDQMRecoGlobalRunCAF           *
    process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripDQMClientGlobalRunCAF         *
    process.qTester                              *
    process.dqmSaver
)

# Input
process.load( "INPUT_FILES" )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)
