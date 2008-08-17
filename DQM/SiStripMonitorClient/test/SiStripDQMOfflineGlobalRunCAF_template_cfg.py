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
# process.load( "Configuration.GlobalRuns.ForceZeroTeslaField_cff" )
process.localUniform = cms.ESProducer( "UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double( 0.0 )
)
process.prefer( "UniformMagneticFieldESProducer" )

# Geometry
process.load( "Configuration.StandardSequences.Geometry_cff" )

# Calibration 
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripCond.toGet = cms.VPSet(
    cms.PSet( record = cms.string( 'SiStripPedestalsRcd') , tag = cms.string( 'SiStripPedestals_TKCC_21X_v3_hlt' )      ), 
    cms.PSet( record = cms.string( 'SiStripNoisesRcd')    , tag = cms.string( 'SiStripNoise_TKCC_21X_v3_hlt' )          ),
    cms.PSet( record = cms.string( 'SiStripBadFiberRcd')  , tag = cms.string( 'SiStripBadChannel_TKCC_21X_v2_offline' ) ),
    cms.PSet( record = cms.string( 'SiStripBadChannelRcd'), tag = cms.string( 'SiStripBadChannel_TKCC_21X_v3_hlt' )     ),
    cms.PSet( record = cms.string( 'SiStripFedCablingRcd'), tag = cms.string( 'SiStripFedCabling_TKCC_21X_v3_hlt' )     )
)
# uncomment for Oracle access at CERN
process.siStripCond.connect                         = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.sistripconn = cms.ESProducer( "SiStripConnectivity" )

process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet( record = cms.string( 'SiStripDetCablingRcd' ), tag = cms.string( '' ) ),
    cms.PSet( record = cms.string( 'SiStripBadChannelRcd' ), tag = cms.string( '' ) )
)

# Fake Conditions
process.load( "CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff" )
process.siStripGainFakeESSource.appendToDataLabel=cms.string('')
process.load( "CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff" )
process.load( "CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff" )
process.load( "CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff" )

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
    process.SiStripDQMRecoGlobalRunCAF           *
    process.SiStripDQMSourceGlobalRunCAF_reduced *
    process.SiStripDQMClientGlobalRunCAF         *
    process.qTester * process.dqmSaver
)

# Input
process.load( "INPUT_FILES" )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)
