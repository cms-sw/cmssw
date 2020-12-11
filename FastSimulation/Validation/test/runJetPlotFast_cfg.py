import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

process.load("FastSimulation.Configuration.QCDpt50_120_cfi")

#  include "FastSimulation/Configuration/data/QCDpt600-800.cfi"
# Generate Minimum Bias Events
#  include "FastSimulation/Configuration/data/MinBiasEvents.cfi"
# Generate muons with a flat pT particle gun, and with pT=10.
# include "FastSimulation/Configuration/data/FlatPtMuonGun.cfi"
# replace FlatRandomPtGunProducer.PGunParameters.PartID={130}
# Generate di-electrons with pT=35 GeV
# include "FastSimulation/Configuration/data/DiElectrons.cfi"
# Famos sequences (no HLT here)
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

#  
# module o1 = PoolOutputModule { 
# untracked string fileName = "MyFirstFamosFile.root" 
# untracked vstring outputCommands = {
# "keep *",
# "drop *_mix_*_*"
# }
# }
# endpath outpath = { o1 }
# 
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.jetComp = cms.EDAnalyzer("JetComparison",
    MinEnergy = cms.double(50.0),
    outputFile = cms.untracked.string('fastjet50-120_fast.root')
)

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.famosWithEverything*process.jetComp)
process.famosPileUp.PileUpSimulator.averageNumber = 0.0
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.fastSimProducer.SimulateCalorimetry = True
for layer in process.fastSimProducer.detectorDefinition.BarrelLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
for layer in process.fastSimProducer.detectorDefinition.ForwardLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.detailedInfo = dict(extension = '.txt')


