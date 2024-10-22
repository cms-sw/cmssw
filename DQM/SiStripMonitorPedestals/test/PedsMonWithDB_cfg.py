import FWCore.ParameterSet.Config as cms

process = cms.Process("PEDESTALS")

# MessageLogger ########
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('PedsMon')
)

# geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# tracker numbering
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# Calibration
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V4P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

# DQM Services
process.DQMStore = cms.Service("DQMStore",
    verbose = cms.untracked.int32(0)
)

#  Reconstruction Modules
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = 'source'

# Pedestal Monitor 
process.load("DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi")
process.load("DQM.SiStripMonitorPedestals.SiStripMonitorRawData_cfi")
process.PedsMon.OutputMEsInRootFile = True

# Input Events
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




