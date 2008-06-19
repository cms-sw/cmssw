import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Generate 20 events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# For histograms
process.load("DQMServices.Core.DQM_cfg")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=180GeV/c2
# include "FastSimulation/Configuration/data/HZZllll.cfi"
#    include "FastSimulation/Configuration/data/QCDpt600-800.cfi"

# Generate muons with a flat pT particle gun, and with pT=10.
#process.load("FastSimulation.Configuration.FlatPtMuonGun_cfi")
#process.FlatRandomPtGunSource.PGunParameters.PartID = [211]
process.load("FastSimulation.Configuration.PythiaTauGun_cfi")

process.test = cms.EDFilter("testEvent",
    # necessary to access true particles 
    ParticleFilter = cms.PSet(
        # All protons with an energy larger with EProton (Gev) are kept
        EProton = cms.double(6000.0),
        # Particles with |eta| > etaMax (momentum direction at primary vertex) 
        # are not simulated 
        etaMax = cms.double(5.0),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        pTMin = cms.double(0.0),
        # Particles with energy smaller than EMin (GeV) are not simulated
        EMin = cms.double(0.0)
    ),
    GeantInfo = cms.bool(False)
)

process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = False
process.famosSimHits.MaterialEffects.PairProduction = False
process.famosSimHits.MaterialEffects.Bremsstrahlung = False
process.famosSimHits.MaterialEffects.EnergyLoss = False
process.famosSimHits.MaterialEffects.MultipleScattering = False
process.famosSimHits.MaterialEffects.NuclearInteraction = False
#process.famosSimHits.ActivateDecays.ActivateDecays = false
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.p = cms.Path(
    process.offlineBeamSpot+
    process.famosPileUp+
    process.famosSimHits+
    process.test
)

