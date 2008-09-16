import FWCore.ParameterSet.Config as cms

process = cms.Process("MonitorDigiRealData")

#--------------------------
# Event Source
#--------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/630/0029CA89-9B71-DD11-8B56-001617C3B6FE.root',
                                      '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/630/02162C6D-9E71-DD11-B740-0016177CA7A0.root',
                                      '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/630/0279F81F-B771-DD11-AD2D-000423D99EEE.root',
                                      '/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/630/0287A430-B071-DD11-9A02-001617E30CD4.root')

)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

#-------------------------------------------------
# Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

#-------------------------------------------------
# Magnetic Field
#-------------------------------------------------
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = 'source'

#--------------------------
# DQM Services
#--------------------------
process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

#--------------------------
# SiStrip MonitorDigi
#--------------------------
process.load("DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi")
process.SiStripMonitorDigi.CreateTrendMEs = True
process.SiStripMonitorDigi.OutputMEsInRootFile = True
process.SiStripMonitorDigi.OutputFileName = 'SiStripMonitorDigi_RealData.root'
process.SiStripMonitorDigi.SelectAllDetectors = True

process.outP = cms.OutputModule("AsciiOutputModule")
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Sequences 
#--------------------------

process.RecoForDQM = cms.Sequence(process.siStripDigis)

process.p = cms.Path(process.RecoForDQM*process.SiStripMonitorDigi)
process.ep = cms.EndPath(process.outP)


