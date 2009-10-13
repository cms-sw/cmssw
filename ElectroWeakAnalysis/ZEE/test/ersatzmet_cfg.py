import FWCore.ParameterSet.Config as cms

process = cms.Process("ErsatzMEtMaker")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1),
   limit = cms.untracked.int32(10000000)
)
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
  #reportEvery = cms.untracked.int32(500),
   reportEvery = cms.untracked.int32(1),
   limit = cms.untracked.int32(10000000)
)
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(True)
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')
process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
#process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
#process.extend(include("RecoEcal/EgammaClusterProducers/data/geometryForClustering.cff"))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

from ElectroWeakAnalysis.ErsatzMEt.Reduced_ZeeFull_cff import newFileNames

process.source = cms.Source("PoolSource",
	fileNames = newFileNames
	#fileNames = cms.untracked.vstring('/store/user/wardrope/Zee/Zee/652884fbfc42ebe755d455783d693c41//Zee_8.root')
   # fileNames = cms.untracked.vstring(
#		'/store/user/wardrope/FirstSkim/Zee_9.root'
 #   )
)

from ElectroWeakAnalysis.ErsatzMEt.ersatzmet_cfi import ErsatzMEtParams 
#from ElectroWeakAnalysis.ErsatzMEt.EtaWeights_cff import EtaWeightsPS 
process.ErsatzMEt = cms.EDAnalyzer('ErsatzMEt',
ErsatzMEtParams,
MCTruthCollection = cms.InputTag("genParticles"),
CaloTowerCollection = cms.InputTag("towerMaker"),
Zevent = cms.bool(False),
mTPmin = cms.double(61.),
mTPmax = cms.double(121.),
etaWidth = cms.int32(7),
phiWidth = cms.int32(25)
)
# Other statements

process.p = cms.Path(process.egammaIsolationSequence*process.ErsatzMEt)
process.TFileService = cms.Service("TFileService",
	fileName = cms.string("Results_ersatz.root"),
)

