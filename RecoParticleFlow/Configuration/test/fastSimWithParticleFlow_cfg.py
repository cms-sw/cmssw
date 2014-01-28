import FWCore.ParameterSet.Config as cms


process = cms.Process("PROD")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

#generation
process.load("RecoParticleFlow.Configuration.source_singleTau_cfi")
#process.load("RecoParticleFlow.Configuration.source_particleGun_cfi")
#process.generator.PGunParameters.ParticleID = cms.vint32(22)
# process.load("FastSimulation.Configuration.SimpleJet_cfi")

#fastsim
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.famosSimHits.VertexGenerator.BetaStar = 0.00001
process.famosSimHits.VertexGenerator.SigmaZ = 0.00001

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

process.p1 = cms.Path(
#    process.famosWithCaloTowersAndParticleFlow +
    process.ProductionFilterSequence +
    process.famosWithEverything +
    process.caloJetMetGen +
    process.particleFlowSimParticle
    )


process.load("FastSimulation.Configuration.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)

process.load("FastSimulation.Configuration.EventContent_cff")
process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('reco.root')
)

process.load("RecoParticleFlow.Configuration.Display_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')
)

process.outpath = cms.EndPath(process.aod + process.display)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(False),
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)
process.MessageLogger.cerr.FwkReport.reportEvery = 10
#
