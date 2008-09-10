import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripDQMFile")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('ctfWithMaterialTracks', 
                                         'SiStripMonitorTrack'),
    cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
    )


#-------------------------------------------------
# CMS Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripCond.toGet = cms.VPSet(
    cms.PSet(record = cms.string('SiStripPedestalsRcd'), tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')), 
    cms.PSet(record = cms.string('SiStripNoisesRcd'), tag = cms.string('SiStripNoise_TKCC_21X_v3_hlt')), 
    cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('SiStripBadChannel_TKCC_21X_v3_hlt')), 
    cms.PSet(record = cms.string('SiStripFedCablingRcd'), tag = cms.string('SiStripFedCabling_TKCC_21X_v3_hlt')))

process.siStripCond.connect = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')), 
    cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('')))

process.load("CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff")

process.load("CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff")

process.load("CalibTracker.Configuration.SiPixelLorentzAngle.SiPixelLorentzAngle_Fake_cff")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")


process.sistripconn = cms.ESProducer("SiStripConnectivity")

#-------------------------------------------------
# DQM
#-------------------------------------------------
# DQMStore Service
process.DQMStore = cms.Service("DQMStore",
                               referenceFileName = cms.untracked.string(''),
                               verbose = cms.untracked.int32(0)
                               )
# SiStripMonitorTrack
process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_WithReco_cff")

#-------------------------------------------------
# Performance Checks
#-------------------------------------------------
# memory
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                        ignoreTotal = cms.untracked.int32(0)
                                        )

# timing
process.Timing = cms.Service("Timing")

#-------------------------------------------------
# In-/Output
#-------------------------------------------------

# input
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/059/0E1028D9-696F-DD11-B244-001617E30E2C.root',
    '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/059/0EB84F22-6A6F-DD11-98DD-001617E30D12.root',
    
    )
                            )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(500))

# output
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('TESTReal.root'),
                               options = cms.PSet(wantSummary = cms.untracked.bool(True))                               
                               )

#-------------------------------------------------
# Scheduling
#-------------------------------------------------
process.dumpinfo = cms.EDAnalyzer("EventContentAnalyzer")

process.outP = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.DQMSiStripMonitorTrack_Real*process.dumpinfo)
process.pout = cms.EndPath(process.out*process.outP)
