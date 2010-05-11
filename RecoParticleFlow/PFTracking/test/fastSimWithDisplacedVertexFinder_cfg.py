import FWCore.ParameterSet.Config as cms


process = cms.Process("PROD")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)

#generation
#process.load("RecoParticleFlow.Configuration.source_singleTau_cfi")
#process.load("RecoParticleFlow.PFTracking.source_jetGun_NuclearTest_cfi")
process.load("RecoParticleFlow.PFTracking.source_particleGun_NuclearTest_cfi")
#process.generator.PGunParameters.ParticleID = cms.vint32(22)
# process.load("FastSimulation.Configuration.SimpleJet_cfi")

#fastsim
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.GlobalTag.globaltag = "MC_37Y_V1::All"


process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.famosSimHits.VertexGenerator.BetaStar = 0.00001
process.famosSimHits.VertexGenerator.SigmaZ = 0.00001

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#process.famosSimHits.MaterialEffects.PairProduction = True
#process.famosSimHits.MaterialEffects.Bremsstrahlung = True
#process.famosSimHits.MaterialEffects.EnergyLoss = True
#process.famosSimHits.MaterialEffects.MultipleScattering = True
#process.famosSimHits.MaterialEffects.NuclearInteraction = True

#process.famosSimHits.MaterialEffects.PairProduction = False
#process.famosSimHits.MaterialEffects.Bremsstrahlung = False
#process.famosSimHits.MaterialEffects.EnergyLoss = False
#process.famosSimHits.MaterialEffects.MultipleScattering = False
#process.famosSimHits.MaterialEffects.NuclearInteraction = False

process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
process.load("RecoParticleFlow.PFTracking.particleFlowDisplacedVertexCandidate_cff")
process.load("RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cff")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.displacedVertexSelector = cms.EDFilter(
    "PFDisplacedVertexSelector",
    src = cms.InputTag("particleFlowDisplacedVertex"),
#    src = cms.InputTag("toto"),
#    cut = cms.string("position.x>1000.0")
    cut = cms.string(""),
    filter = cms.bool(True)
    )

process.simVertexSelector = cms.EDFilter(
    "SimVertexSelector",
    src = cms.InputTag("famosSimHits"),
#    cut = cms.string("position.x>1000.0")
    cut = cms.string("position.rho>2.5"),
#    cut = cms.string(""),
    filter = cms.bool(True)
    )


#process.selector = cms.Path(process.displacedVertexSelector*process.simVertexSelector)
process.selector = cms.Path(process.simVertexSelector)

process.particleFlow.rejectTracks_Bad =  cms.bool(True)
process.particleFlow.rejectTracks_Step45 = cms.bool(True)
process.particleFlowBlock.useNuclear = cms.bool(True)
process.particleFlowBlock.useConversions = cms.bool(False)
process.particleFlowBlock.useV0 = cms.bool(False)

process.particleFlow.usePFNuclearInteractions = cms.bool(True)
process.particleFlow.usePFConversions = cms.bool(False)
process.particleFlow.usePFDecays = cms.bool(False)
#process.particleFlow.correctPrimary = cms.bool(True)
#process.particleFlow.correctSecondary = cms.bool(True)


process.p1 = cms.Path(
#    process.famosWithCaloTowersAndParticleFlow +
    process.ProductionFilterSequence +
    process.famosWithEverything +# +
    process.caloJetMetGen +
    process.particleFlowSimParticle
#    process.particleFlowDisplacedVertexCandidate +
#    process.particleFlowDisplacedVertex
    
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
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("selector")),
    fileName = cms.untracked.string('display.root')
)

process.outpath = cms.EndPath(process.display)

process.schedule = cms.Schedule(
    process.p1,
    process.selector
#    process.outpath
)



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
