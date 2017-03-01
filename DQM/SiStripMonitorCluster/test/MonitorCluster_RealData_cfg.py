import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineRealData")
#--------------------------
# Event Source
#--------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/data/Run2010B/MinimumBias/RAW/v1/000/149/011/0042F6EF-0AE1-DF11-9237-003048F1C420.root'
    'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root'
    #'/store/data/Run2011A/Jet/RAW/v1/000/173/692/18F5E657-E6CC-E011-97EE-BCAEC54DB5D6.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

#-------------------------------------------------
# Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siStripDigis',
                                         'siStripZeroSuppression',
                                         'siStripClusters'
                                         'SiStripMonitorCluster'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

#-----------------------------
# Magnetic Field
#-----------------------------
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#-------------------------------------------------
# Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_44_V4::All"
#process.GlobalTag.globaltag = "GR_P_V22::All"

#-----------------------
#  Reconstruction Modules
#-----------------------
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
#process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
#process.siStripDigis.ProductLabel = 'source'

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

#--------------------------
# DQM Services
#--------------------------
process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)


process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

#--------------------------
# Producers
#--------------------------
# Event History Producer
process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

# APV Phase Producer
#nwe one
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts_cfi")
from DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi import *
#APVPhases = cms.EDProducer("APVCyclePhaseProducerFromL1TS",
#                            defaultPartitionNames = cms.vstring("TI",
#                            "TO",
#                            "TP",
#                            "TM"
#                            ),
#                            defaultPhases = cms.vint32(60,60,60,60),
#                            magicOffset = cms.untracked.int32(258),
#                            l1TSCollection = cms.InputTag("scalersRawToDigi"),
#                            )

#--------------------------
# SiStrip MonitorCluster
#--------------------------
process.load("DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi")
process.SiStripMonitorCluster.CreateTrendMEs = True
process.SiStripMonitorCluster.TkHistoMap_On = True
process.SiStripMonitorCluster.SelectAllDetectors = True
process.SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True
process.SiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon = True
process.SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True
process.SiStripMonitorCluster.TH2ClustersApvCycle.subdetswitchon = True
process.SiStripMonitorCluster.TH2ClustersApvCycle.yfactor = 0.005
process.SiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon=True
process.SiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon=True
process.SiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon=True
process.SiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon=True
process.SiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon=True


process.outP = cms.OutputModule("AsciiOutputModule")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.RecoForDQM = cms.Sequence(process.siPixelDigis*process.siPixelClusters*process.siStripDigis*process.gtDigis*process.siStripZeroSuppression*process.siStripClusters)
process.p = cms.Path(
    process.scalersRawToDigi*
    process.APVPhases*
    process.consecutiveHEs*
    process.RecoForDQM*process.SiStripMonitorCluster)
process.ep = cms.EndPath(process.outP)

