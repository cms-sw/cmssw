import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')

process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring("*"),
                                    log = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('log')
                                    )


process.TFileService =  cms.Service( "TFileService",
                                     fileName = cms.string(""),
                                     closeFileFast = cms.untracked.bool(True)
                                     )

process.source = cms.Source ( "PoolSource",
                              fileNames = cms.untracked.vstring(""),
                              secondaryFileNames = cms.untracked.vstring()
                              )


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")
process.load('Configuration/EventContent/EventContent_cff')

#------------------------------------------
# Load ClusterToDigiProducer
#------------------------------------------


process.load('RecoLocalTracker.SiStripClusterizer.SiStripClusterToDigiProducer_cfi')

process.siStripClustersNew = process.siStripClusters.clone()
for inTag in process.siStripClustersNew.DigiProducersList:
    inTag.moduleLabel = "siStripClustersToDigis"

process.siStripClusters.Clusterizer.RemoveApvShots    = cms.bool(False)
process.siStripClustersNew.Clusterizer.RemoveApvShots = cms.bool(True)

#------------------------------------------
# Load DQM 
#------------------------------------------

process.DQMStore = cms.Service("DQMStore")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi")

process.SiStripMonitorClusterNew = process.SiStripMonitorCluster.clone()
process.SiStripMonitorClusterNew.TopFolderName = "ClusterToDigi"
process.SiStripMonitorClusterNew.ClusterProducerStrip = 'siStripClustersNew'

print process.SiStripMonitorClusterNew.ClusterProducerStrip
print process.SiStripMonitorCluster.ClusterProducerStrip

process.SiStripMonitorCluster.TH1ClusterWidth.Nbinx          = cms.int32(129)
process.SiStripMonitorCluster.TH1ClusterWidth.xmax           = cms.double(128.5)
process.SiStripMonitorCluster.TH1ClusterDigiPos.moduleswitchon= cms.bool(True)


process.SiStripMonitorClusterNew.TH1ClusterWidth.Nbinx          = cms.int32(129)
process.SiStripMonitorClusterNew.TH1ClusterWidth.xmax           = cms.double(128.5)
process.SiStripMonitorClusterNew.TH1ClusterDigiPos.moduleswitchon= cms.bool(True)
#------------------------------------------
# Load apvshotanalyzer
#------------------------------------------

process.load("DPGAnalysis.SiStripTools.apvshotsanalyzer_cfi")
process.apvshotsanalyzer.digiCollection.moduleLabel = "siStripClustersToDigis" 

#------------------------------------------
# Paths
#------------------------------------------


process.skimming = cms.EDFilter("FilterOutScraping",
                                applyfilter = cms.untracked.bool(True),
                                debugOn = cms.untracked.bool(True),
                                numtrack = cms.untracked.uint32(10),
                                thresh = cms.untracked.double(0.2)
                                )

process.outpath = cms.EndPath(process.skimming+
                              process.siStripDigis+
                              process.siStripZeroSuppression+
                              process.siStripClusters+
                              process.SiStripMonitorCluster+
                              process.siStripClustersToDigis+
                              process.siStripClustersNew+
                              process.SiStripMonitorClusterNew+
                              process.apvshotsanalyzer
                              ) 


#---------------------------------------------------------
# Run Dependent Configuration
#---------------------------------------------------------

#process.GlobalTag.globaltag = 'GR09_R_34X_V3::All'
process.GlobalTag.globaltag = 'GR_R_44_V14::All'

process.TFileService.fileName = cms.string("apvShotsAnalysis.root")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

process.source.fileNames = cms.untracked.vstring(
"/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/00A7606D-22F9-E011-8437-001A92971BDC.root"
)
