import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripDQMOfflineGlobalRunCAF")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO') ),
    destinations = cms.untracked.vstring('cout')
)

# Magnetic Field
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

# Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# Calibration 
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripCond.toGet = cms.VPSet(
    cms.PSet( record = cms.string('SiStripPedestalsRcd'),  tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')), 
    cms.PSet( record = cms.string('SiStripNoisesRcd'),     tag = cms.string('SiStripNoise_TKCC_21X_v3_hlt')),
    cms.PSet( record = cms.string('SiStripBadChannelRcd'), tag = cms.string('SiStripBadChannel_TKCC_21X_v3_hlt')),
    cms.PSet( record = cms.string('SiStripFedCablingRcd'), tag = cms.string('SiStripFedCabling_TKCC_21X_v3_hlt')) )

process.siStripCond.connect = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet( record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')),
    cms.PSet( record = cms.string('SiStripBadChannelRcd'), tag = cms.string('')) )

process.load("CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff")
process.load("CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff")
process.load("CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff")
process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

# SISTRIP DQM
##process.load("INCLUDE_DIRECTORY.SiStripDQMOfflineGlobalRunCAF_cff")

# HLT Filter
process.hltFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLTTrackerCosmicsCoTF', 'CandHLTTrackerCosmicsRS', 'CandHLTTrackerCosmicsCTF'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","FU"))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1) )

# Scheduling
process.p = cms.Path(process.SiStripDQMRecoGlobalRunCAF*process.SiStripDQMSourceGlobalRunCAF_reduced*process.SiStripDQMClientGlobalRunCAF*process.qTester*process.dqmSaver)
