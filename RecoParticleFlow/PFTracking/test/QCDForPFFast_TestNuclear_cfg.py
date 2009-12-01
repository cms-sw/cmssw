import FWCore.ParameterSet.Config as cms


process = cms.Process("PROD")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#generation
process.source = cms.Source("EmptySource")
process.load("Configuration.Generator.QCDForPF_cfi")
process.generator.comEnergy = 14000.

#fastsim
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

process.RandomNumberGeneratorService.generator.initialSeed=1
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

# Get frontier conditions    - not applied in the HCAL, see below
# Values for globaltag are "STARTUP_V5::All", "1PB::All", "10PB::All", "IDEAL_V5::All"
process.GlobalTag.globaltag = "MC_3XY_V14::All"

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# process.famosSimHits.MaterialEffects.PairProduction = False
# process.famosSimHits.MaterialEffects.Bremsstrahlung = False
# process.famosSimHits.MaterialEffects.EnergyLoss = False
# process.famosSimHits.MaterialEffects.MultipleScattering = False
# process.famosSimHits.MaterialEffects.NuclearInteraction = False

process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
process.load("RecoParticleFlow.PFTracking.particleFlowDisplacedVertexCandidate_cff")
process.load("RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cff")


process.p1 = cms.Path(
    process.generator +
    process.famosWithEverything +
    process.caloJetMetGen +
    process.particleFlowSimParticle +
    process.particleFlowDisplacedVertexCandidate +
    process.particleFlowDisplacedVertex
)


process.load("FastSimulation.Configuration.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)

process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('reco.root')
)

process.load("RecoParticleFlow.Configuration.Display_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')
)

#process.outpath = cms.EndPath(process.aod + process.reco + process.display)
process.outpath = cms.EndPath(process.aod+process.display)

#
