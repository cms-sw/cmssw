import FWCore.ParameterSet.Config as cms

ecalOfflineCosmicClient = cms.EDAnalyzer("EcalOfflineCosmicClient",
   fileName = cms.untracked.string("EcalOfflineCosmicTaskClient.root"),
   saveFile = cms.untracked.bool(False),
   endFunction = cms.string("endLuminosityBlock"),
   rootDir = cms.string("EcalOfflineCosmicTask"),
   subDetDirs = cms.vstring("EEM","EB","EEP"),
   l1TriggerDirs = cms.vstring("AllEvents","CSC","DT","ECAL","HCAL","RPC"),
   timingDir = cms.string("TimingHists"),
   timingTTBinned = cms.string("timingTTBinning3D"),
   timingModBinned = cms.string("timingModBinning3D"),
   timingVsAmp = cms.string("timingVsAmp2D"),
   clusterDir = cms.string("ClusterHists"),
   clusterPlots = cms.vstring("NumBCinSCphi2D","NumXtalsVsEnergy2D")
)


