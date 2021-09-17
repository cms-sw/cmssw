# The following comments couldn't be translated into the new config version:

#--------------------------
# DQM Services
#--------------------------

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineSimData")

#--------------------------
# Event Source
#--------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_2_6/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/F6CBADA1-D29A-DE11-96C8-00304879FA4A.root',
    '/store/relval/CMSSW_3_2_6/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/60372DB3-D09A-DE11-857C-001D09F2915A.root')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

#-------------------------------------------------
# Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring(
        'siStripDigis', 
        'siStripZeroSuppression', 
        'siStripClustersSiStripMonitorCluster'
    )
)

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# CALIBRATION
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "STARTUP31X_V7::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = 'rawDataCollector'

#process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_SimData_cfi")
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

#--------------------------
# DQM Services
#--------------------------
process.load("DQM.SiStripCommon.TkHistoMap_cff")

#--------------------------
# SiStrip MonitorCluster
#--------------------------
process.load("DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi")
process.SiStripMonitorCluster.CreateTrendMEs = True
process.SiStripMonitorCluster.TkHistoMap_On = True
process.SiStripMonitorCluster.SelectAllDetectors = True
#process.SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True
process.SiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon = True
#process.SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True

##
process.outP = cms.OutputModule("AsciiOutputModule")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.RecoForDQM = cms.Sequence(process.siStripDigis*process.siStripZeroSuppression*process.siStripClusters)
process.p = cms.Path(process.RecoForDQM*process.SiStripMonitorCluster)
process.ep = cms.EndPath(process.outP)

