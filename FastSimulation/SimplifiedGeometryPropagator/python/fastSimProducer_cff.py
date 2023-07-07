import FWCore.ParameterSet.Config as cms

from FastSimulation.Event.ParticleFilter_cfi import  ParticleFilterBlock
from FastSimulation.SimplifiedGeometryPropagator.TrackerMaterial_cfi import TrackerMaterialBlock
from FastSimulation.SimplifiedGeometryPropagator.CaloMaterial_cfi import CaloMaterialBlock # Hack to interface "old" calorimetry with "new" propagation in tracker
from FastSimulation.Calorimetry.Calorimetry_cff import *
from FastSimulation.MaterialEffects.MaterialEffects_cfi import *

fastSimProducer = cms.EDProducer(
    "FastSimProducer",
    src = cms.InputTag("generatorSmeared"),
    particleFilter =  ParticleFilterBlock.ParticleFilter,
    trackerDefinition = TrackerMaterialBlock.TrackerMaterial,
    simulateCalorimetry = cms.bool(True),
    simulateMuons = cms.bool(True),
    caloDefinition = CaloMaterialBlock.CaloMaterial, #  Hack to interface "old" calorimetry with "new" propagation in tracker
    beamPipeRadius = cms.double(3.),
    deltaRchargedMother = cms.double(0.02), # Maximum angle to associate a charged daughter to a charged mother (mostly done to associate muons to decaying pions)
    interactionModels = cms.PSet(
            pairProduction = cms.PSet(
                className = cms.string("fastsim::PairProduction"),
                photonEnergyCut = cms.double(0.1),
                # silicon
                Z = cms.double(14.0000)
                ),
            nuclearInteraction = cms.PSet(
                className = cms.string("fastsim::NuclearInteraction"),
                distCut = cms.double(0.020),
                hadronEnergy = cms.double(0.2), # the smallest momentum for elastic interactions
                # inputFile = cms.string("NuclearInteractionInputFile.txt"), # the file to read the starting interaction in each files (random reproducibility in case of a crash)
                ),
            #nuclearInteractionFTF = cms.PSet(
            #    className = cms.string("fastsim::NuclearInteractionFTF"),
            #    distCut = cms.double(0.020),
            #    bertiniLimit = cms.double(3.5), # upper energy limit for the Bertini cascade 
            #    energyLimit = cms.double(0.1), # Kinetic energy threshold for secondaries 
            #    ),
            bremsstrahlung = cms.PSet(
                className = cms.string("fastsim::Bremsstrahlung"),
                minPhotonEnergy = cms.double(0.1),
                minPhotonEnergyFraction = cms.double(0.005),
                # silicon
                Z = cms.double(14.0000)
                ),
            #muonBremsstrahlung = cms.PSet(
            #    className = cms.string("fastsim::MuonBremsstrahlung"),
            #    minPhotonEnergy = cms.double(0.1),
            #    minPhotonEnergyFraction = cms.double(0.005),
            #    # silicon
            #    A = cms.double(28.0855),
            #    Z = cms.double(14.0000),
            #    density = cms.double(2.329),
            #    radLen = cms.double(9.360)
            #    ),
            energyLoss = cms.PSet(
                className = cms.string("fastsim::EnergyLoss"),
                minMomentumCut = cms.double(0.1),
                # silicon
                A = cms.double(28.0855),
                Z = cms.double(14.0000),
                density = cms.double(2.329),
                radLen = cms.double(9.360)
                ),
            multipleScattering = cms.PSet(
                className = cms.string("fastsim::MultipleScattering"),
                minPt = cms.double(0.2),
                # silicon
                radLen = cms.double(9.360)
                ),
            trackerSimHits = cms.PSet(
                className = cms.string("fastsim::TrackerSimHitProducer"),
                minMomentumCut = cms.double(0.1),
                doHitsFromInboundParticles = cms.bool(False), # Track reconstruction not possible for those particles so hits do not have to be simulated
                ),    
        ),
    Calorimetry = FamosCalorimetryBlock.Calorimetry,
    MaterialEffectsForMuonsInECAL = MaterialEffectsForMuonsInECALBlock.MaterialEffectsForMuonsInECAL,
    MaterialEffectsForMuonsInHCAL = MaterialEffectsForMuonsInHCALBlock.MaterialEffectsForMuonsInHCAL,
    GFlash = FamosCalorimetryBlock.GFlash,
    fixLongLivedBug = cms.bool(False),
    useFastSimsDecayer = cms.bool(True),
)

from Configuration.ProcessModifiers.fastSimFixLongLivedBug_cff import fastSimFixLongLivedBug
fastSimFixLongLivedBug.toModify(fastSimProducer, fixLongLivedBug = cms.bool(True))

from Configuration.ProcessModifiers.useGenNotFastSimDecays_cff import useGenNotFastSimDecays
useGenNotFastSimDecays.toModify(fastSimProducer, useFastSimsDecayer = cms.bool(False))


