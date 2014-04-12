import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalysisErsatz")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1),
  limit = cms.untracked.int32(10000000)
)
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
 # reportEvery = cms.untracked.int32(1000),
   reportEvery = cms.untracked.int32(1),
   limit = cms.untracked.int32(10000000)
)
 
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

#from ElectroWeakAnalysis.ErsatzMEt.Wenu_cff import newFileNames
#from ElectroWeakAnalysis.AnalysisErsatz.Wenu_Lots_cff import newFileNames
#from ElectroWeakAnalysis.ErsatzMEt.Reduced_Zee_cff import newFileNames

process.source = cms.Source("PoolSource",
	#fileNames = newFileNames
    # replace 'myfile.root' with the source file you want to use
#	fileNames = cms.untracked.vstring()
    fileNames = cms.untracked.vstring("file:/tmp/rnandi/WenuTrue_AODSIM.root")
)

# Other statements
process.WenuMcEleFilter = cms.EDFilter("PythiaFilter",
    Status = cms.untracked.int32(1),
    MotherID = cms.untracked.int32(11),
    MinPt = cms.untracked.double(0.0),
    ParticleID = cms.untracked.int32(11),
    MaxEta = cms.untracked.double(2.7),
    MinEta = cms.untracked.double(-2.7)
)
process.WenuFilSeq = cms.Sequence(process.WenuMcEleFilter)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string("STARTUP31X_V4::All")
process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
#process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
#process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequencePAT_cff")

from ElectroWeakAnalysis.ZEE.ersatzmet_cfi import ErsatzMEtParams

process.analyse = cms.EDAnalyzer('AnalysisErsatz',
ErsatzMEtParams,
ErsatzEvent = cms.bool(False)
)

process.add_( cms.Service(
	"RandomNumberGeneratorService", 
	analyse = cms.PSet(
				initialSeed = cms.untracked.uint32(563545),
				engineName = cms.untracked.string('TRandom3')) 
))
#process.p = cms.Path(process.WenuFilSeq*process.analyse)
#process.p = cms.Path(process.egammaIsolationSequence*process.analyse)
process.p = cms.Path(process.analyse)
#process.p = cms.Path(process.WenuFilSeq*process.egammaIsolationSequence*process.analyse)
process.TFileService = cms.Service("TFileService",
        fileName = cms.string("WenuTrue_AODSIM.root"),
)
