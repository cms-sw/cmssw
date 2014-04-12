import FWCore.ParameterSet.Config as cms


process = cms.Process("PROD")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

#generation
# process.load("RecoParticleFlow.Configuration.source_singleTau_cfi")
# process.load("FastSimulation.Configuration.SimpleJet_cfi")

from Configuration.Generator.PythiaUESettings_cfi import *
process.source = cms.Source(
    "PythiaSource",
    pythiaVerbosity = cms.untracked.bool(False),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(1),
    DoubleParticle = cms.untracked.bool(True),
    Ptmin = cms.untracked.double(20.0),
    Ptmax = cms.untracked.double(700.0),
#    Emin = cms.untracked.double(10.0),
#    Emax = cms.untracked.double(10.0),
    Etamin = cms.untracked.double(0.0),
    Etamax = cms.untracked.double(1.0),
    Phimin = cms.untracked.double(0.0),
    Phimax = cms.untracked.double(360.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        # Tau jets only
        pythiaJets = cms.vstring(),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaUESettings',
            'pythiaJets'
        )

    )
    
)

#fastsim
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.famosSimHits.VertexGenerator.BetaStar = 0.00001
process.famosSimHits.VertexGenerator.SigmaZ = 0.00001

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# process.famosSimHits.MaterialEffects.PairProduction = false
# process.famosSimHits.MaterialEffects.Bremsstrahlung = false
# process.famosSimHits.MaterialEffects.EnergyLoss = false
# process.famosSimHits.MaterialEffects.MultipleScattering = false
# process.famosSimHits.MaterialEffects.NuclearInteraction = false

#process.load("RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff")

process.pfTaus = cms.EDFilter("PFTauSelector",
    src = cms.InputTag("pfRecoTauProducer"),
    discriminator = cms.InputTag("pfRecoTauDiscriminationByIsolation")
)

process.p1 = cms.Path(
    process.famosWithEverything *
    process.caloJetMetGen +
    process.pfTaus
    )


process.load("FastSimulation.Configuration.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)
process.aod.outputCommands.append('keep edmHepMCProduct_*_*_*')
process.aod.outputCommands.append('keep recoPFTaus_*_*_*')

process.outpath = cms.EndPath(process.aod)

#
