# The following comments couldn't be translated into the new config version:

#for oracle access at cern uncomment

#--------------------------
# DQM Services
#--------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineSimData")
#-------------------------------------------------
# MAGNETIC FIELD
#-------------------------------------------------
# Magnetic fiuld: force mag field to be 0.0 tesla
#    include "Configuration/GlobalRuns/data/ForceZeroTeslaField.cff"
# tracker geometry
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

# tracker numbering
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# cms geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")

#--------------------------
# SiStrip MonitorDigi
#--------------------------
process.load("DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
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
    fileNames = cms.untracked.vstring('/store/data/CRUZET3/Cosmics/RAW/v1/000/051/490/02D59D05-4151-DD11-9E79-001617DBD5AC.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.RecoForDQM = cms.Sequence(process.siStripDigis)
process.p = cms.Path(process.RecoForDQM*process.SiStripMonitorDigi)
process.ep = cms.EndPath(process.outP)
process.siStripCond.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripPedestalsRcd'),
    tag = cms.string('SiStripPedestals_TKCC_20X_v3_hlt')
), 
    cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoise_TKCC_20X_v3_hlt')
    ), 
    cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripFedCabling_TKCC_20X_v3_hlt')
    ))
process.siStripCond.connect = 'oracle://cms_orcoff_prod/CMS_COND_20X_STRIP'
process.siStripCond.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.siStripDigis.ProductLabel = 'source'
process.SiStripMonitorDigi.CreateTrendMEs = True
process.SiStripMonitorDigi.OutputMEsInRootFile = True
process.SiStripMonitorDigi.OutputFileName = 'SiStripMonitorDigi_RealData.root'
process.SiStripMonitorDigi.SelectAllDetectors = True

