# The following comments couldn't be translated into the new config version:

#for oracle access at cern uncomment

#--------------------------
# DQM Services
#--------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineSimData")
# tracker geometry
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

# tracker numbering
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# cms geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")

process.load("CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff")

process.load("CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff")

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff")

#--------------------------
# SiStrip MonitorCluster
#--------------------------
process.load("DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis', 
        'SiStripMonitorDigi'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

process.outP = cms.OutputModule("AsciiOutputModule")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/02D59D05-4151-DD11-9E79-001617DBD5AC.root', 
        '/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/02E220B3-4451-DD11-8471-000423D98868.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)
process.RecoForDQM = cms.Sequence(process.siStripDigis*process.siStripZeroSuppression*process.siStripClusters)
process.p = cms.Path(process.RecoForDQM*process.SiStripMonitorCluster)
process.ep = cms.EndPath(process.outP)
process.siStripCond.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripPedestalsRcd'),
    tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')
), 
    cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoise_TKCC_21X_v3_hlt')
    ), 
    cms.PSet(
        record = cms.string('SiStripBadChannelRcd'),
        tag = cms.string('SiStripBadChannel_TKCC_21X_v3_hlt')
    ), 
    cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripFedCabling_TKCC_21X_v3_hlt')
    ))
process.siStripCond.connect = 'oracle://cms_orcoff_prod/CMS_COND_21X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(cms.PSet(
    record = cms.string('SiStripDetCablingRcd'),
    tag = cms.string('')
), 
    cms.PSet(
        record = cms.string('SiStripBadChannelRcd'),
        tag = cms.string('')
    ))
process.siStripDigis.ProductLabel = 'source'
process.SiStripMonitorCluster.OutputMEsInRootFile = True
process.SiStripMonitorCluster.SelectAllDetectors = True
process.SiStripMonitorCluster.OutputFileName = 'SiStripMonitorCluster.root'

