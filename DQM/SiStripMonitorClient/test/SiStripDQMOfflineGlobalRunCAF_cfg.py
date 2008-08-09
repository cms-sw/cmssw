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
process.load("DQM.SiStripMonitorClient.SiStripDQMOfflineGlobalRunCAF_cff")

# HLT Filter
process.hltFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLTTrackerCosmicsCoTF', 'CandHLTTrackerCosmicsRS', 'CandHLTTrackerCosmicsCTF'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","FU") )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/1037398E-2551-DD11-B1B0-000423D98FBC.root', 
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/10BED589-2951-DD11-B71D-001617E30CE8.root', 
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/2C1A7C89-2951-DD11-9606-000423D9939C.root', 
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/2E1D548D-2551-DD11-AB4B-000423D987E0.root', 
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/32B92C6E-2651-DD11-A97F-000423D95220.root', 
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/3419CC91-2951-DD11-8A03-000423D6C8EE.root', 
        '/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3/0060/3A33B5CC-2B51-DD11-919A-001617DBCF1E.root'))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000) )

process.p = cms.Path(process.hltFilter*process.SiStripDQMRecoGlobalRunCAF*process.SiStripDQMSourceGlobalRunCAF_reduced*process.SiStripDQMClientGlobalRunCAF*process.qTester*process.dqmSaver)

