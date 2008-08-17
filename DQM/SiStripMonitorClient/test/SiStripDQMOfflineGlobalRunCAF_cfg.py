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

# Geometry
process.load( "Configuration.StandardSequences.Geometry_cff" )

# Calibration 
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripCond.toGet = cms.VPSet(
    cms.PSet( record = cms.string( 'SiStripPedestalsRcd' ) , tag = cms.string( 'SiStripPedestals_TKCC_21X_v3_hlt' )      ), 
    cms.PSet( record = cms.string( 'SiStripNoisesRcd' )    , tag = cms.string( 'SiStripNoise_TKCC_21X_v3_hlt' )          ),
    cms.PSet( record = cms.string( 'SiStripBadFiberRcd' )  , tag = cms.string( 'SiStripBadChannel_TKCC_21X_v2_offline' ) ),
    cms.PSet( record = cms.string( 'SiStripBadChannelRcd' ), tag = cms.string( 'SiStripBadChannel_TKCC_21X_v3_hlt' )     ),
    cms.PSet( record = cms.string( 'SiStripFedCablingRcd' ), tag = cms.string( 'SiStripFedCabling_TKCC_21X_v3_hlt' )     )
)
# uncomment for Oracle access at CERN
process.siStripCond.connect                         = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.sistripconn = cms.ESProducer( "SiStripConnectivity" )

process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet( record = cms.string( 'SiStripDetCablingRcd' ), tag = cms.string( '' ) ),
    cms.PSet( record = cms.string( 'SiStripBadChannelRcd' ), tag = cms.string( '' ) ),
    cms.PSet( record = cms.string( 'SiStripBadFiberRcd' )  , tag = cms.string( '' ) )
)

# Fake Conditions
process.load( "CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff" )
process.siStripGainFakeESSource.appendToDataLabel=cms.string('')
process.load( "CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff" )
process.load( "CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff" )
process.load( "CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff" )

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
