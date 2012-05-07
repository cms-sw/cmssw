import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineRealData")
#--------------------------
# Event Source
#--------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Run2010B/MinimumBias/RAW/v1/000/149/011/0042F6EF-0AE1-DF11-9237-003048F1C420.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
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
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_311_V1::All"

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
APVPhases = cms.EDProducer("APVCyclePhaseProducerFromL1TS",
                            defaultPartitionNames = cms.vstring("TI",
                            "TO",
                            "TP",
                            "TM"
                            ),
                            defaultPhases = cms.vint32(60,60,60,60),
                            magicOffset = cms.untracked.int32(258),
                            l1TSCollection = cms.InputTag("scalersRawToDigi"),
                            )

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
process.SiStripMonitorCluster.OutputMEsInRootFile = True
process.SiStripMonitorCluster.OutputFileName = 'SiStripMonitorCluster.root'


process.outP = cms.OutputModule("AsciiOutputModule")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.RecoForDQM = cms.Sequence(process.siStripDigis*process.gtDigis*process.siStripZeroSuppression*process.siStripClusters)
process.p = cms.Path(
    process.scalersRawToDigi*
    process.APVPhases*
    process.consecutiveHEs*
    process.RecoForDQM*process.SiStripMonitorCluster)
process.ep = cms.EndPath(process.outP)

