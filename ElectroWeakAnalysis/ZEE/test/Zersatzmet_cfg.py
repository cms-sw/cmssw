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
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All')
process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
process.electronHcalTowerIsolationLcone.intRadius = 0.0
#process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff")
#process.extend(include("RecoEcal/EgammaClusterProducers/data/geometryForClustering.cff"))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

#from ElectroWeakAnalysis.ErsatzMEt.Zee_data_cff import newFileNames

process.source = cms.Source("PoolSource",
#	fileNames = newFileNames
    fileNames = cms.untracked.vstring()
    #fileNames = cms.untracked.vstring("file:/tmp/rnandi/Zee_AODSIM.root", "file:/tmp/rnandi/Zee_AODSIM_2.root")
#	"/store/mc/Summer08/Zee/GEN-SIM-RECO/IDEAL_V11_redigi_v2/0008/0840D3A9-A000-DE11-ABF8-00161725E4EB.root")
)

from ElectroWeakAnalysis.ZEE.ersatzmet_cfi import ErsatzMEtParams 
#from ElectroWeakAnalysis.ErsatzMEt.EtaWeights_cff import EtaWeightsPS 
process.ErsatzMEt = cms.EDAnalyzer('ErsatzMEt',
ErsatzMEtParams,
Zevent = cms.bool(True),
HLTPathCheck = cms.bool(False)
)
# Other statements
process.ZeeMcEleFilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(3, 3),
    MinPt = cms.untracked.vdouble(0.0, 0.0),
    MaxEta = cms.untracked.vdouble(2.7, 2.7),
    MinEta = cms.untracked.vdouble(-2.7, -2.7),
    ParticleCharge = cms.untracked.int32(0),
    ParticleID1 = cms.untracked.vint32(11),
    ParticleID2 = cms.untracked.vint32(11)
)
process.ZeeFilSeq = cms.Sequence(process.ZeeMcEleFilter)

#process.p = cms.Path(process.ZeeFilSeq*process.egammaIsolationSequence*process.corMetType1Icone5*process.ErsatzMEt)
#process.p = cms.Path(process.ZeeFilSeq*process.egammaIsolationSequence*process.ErsatzMEt)
process.p = cms.Path(process.ErsatzMEt)
process.TFileService = cms.Service("TFileService", fileName = cms.string("Zee_AODSIM.root"))
