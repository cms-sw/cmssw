import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMOffline.EGamma.zmumugammaAnalyzer_cfi")
process.load("DQMOffline.EGamma.photonOfflineClient_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")

DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	
	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/0EFD3E1B-3D98-E111-AB78-002618FDA211.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/1210DFA2-3098-E111-8D18-00261894382D.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/1294263F-2898-E111-930C-003048678FD6.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/12AD5991-4D98-E111-880F-002618943860.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/18409BDE-4D98-E111-B43A-001A92810AE0.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/1A8127B9-4E98-E111-9076-0026189438C9.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/1AB69670-5C98-E111-81DC-003048678F9C.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/20B397A7-3598-E111-B4CF-002618943950.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/228DCD00-4D98-E111-B83E-003048D15E14.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/22F2F7E8-3998-E111-B04B-0026189438DD.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/004CA74B-3198-E111-9786-003048FF9AC6.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/2879A188-3198-E111-AFD5-0026189438D3.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/2A9304D4-6198-E111-824A-00304867BFA8.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/2C007199-2598-E111-85DE-002618943947.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/2E74EAE7-5A98-E111-8650-002618FDA248.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/02A11927-2D98-E111-902E-002354EF3BD0.root',
##	'/store/mc/Summer12/DYToMuMu_M-20_CT10_TuneZ2star_8TeV-powheg-pythia6/AODSIM/PU_S8_START52_V9-v2/0000/343A7770-7998-E111-8CD0-00304867C1BC.root',


))

from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)
#photonAnalysis.standAlone = cms.bool(True)

from DQMOffline.EGamma.photonOfflineClient_cfi import *
photonOfflineClient.standAlone = cms.bool(True)


#from DQMServices.Components.DQMStoreStats_cfi import *
#dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
#dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


#process.p1 = cms.Path(process.photonAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)
process.p1 = cms.Path(process.photonAnalysis*process.zmumugammaAnalysis*process.photonOfflineClient*process.dqmStoreStats)
#process.p1 = cms.Path(process.photonAnalysis*process.photonOfflineClient*process.dqmStoreStats)


process.schedule = cms.Schedule(process.p1)

