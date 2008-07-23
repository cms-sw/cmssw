# The following comments couldn't be translated into the new config version:

# MessageLogger ########

#for oracle access at cern uncomment

#--------------------------
# DQM Services
#--------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("PEDESTALS")
# tracker geometry
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

# tracker numbering
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# cms geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")

#---------------------------
# Pedestal Monitor 
#---------------------------
process.load("DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi")

process.load("DQM.SiStripMonitorPedestals.SiStripMonitorRawData_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('PedsMon'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('Error')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

process.outPrint = cms.OutputModule("AsciiOutputModule")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/d/dutta/TKCC/49962/USC.00049962.0001.A.storageManager.0.0000.root', 
        'rfio:/castor/cern.ch/user/d/dutta/TKCC/49962/USC.00049962.0002.A.storageManager.0.0000.root', 
        'rfio:/castor/cern.ch/user/d/dutta/TKCC/49962/USC.00049962.0003.A.storageManager.0.0000.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.RecoForDQM = cms.Sequence(process.siStripDigis)
process.p = cms.Path(process.RecoForDQM*process.RawDataMon*process.PedsMon)
process.ep = cms.EndPath(process.outPrint)
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
process.PedsMon.OutputMEsInRootFile = True

