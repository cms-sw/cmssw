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
    useFastSimDecayer = cms.bool(False),
    verboseDecayer = cms.bool(False),
    caloDefinition = CaloMaterialBlock.CaloMaterial, #  Hack to interface "old" calorimetry with "new" propagation in tracker
    beamPipeRadius = cms.double(3.),
    # CloseByParticleGun support: when > 0 [cm], a primary accepted via
    # particleFilter.acceptCaloVertices is moved backwards along its momentum by
    # this distance before the calo-layer navigation, so a vertex sitting on or
    # just past the entrance layer is still picked up. 0 disables the shift.
    caloVertexBackupDistance = cms.double(0.),
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
                saveOutput = cms.untracked.bool(False),
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
)

from Configuration.Eras.Modifier_phase2_fastSim_cff import phase2_fastSim
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal

# Phase-2 calorimetry used to be switched off wholesale because the Run-2 endcap
# ECAL/preshower geometry it assumed is null in Phase-2 and the job segfaulted.
# With the HGCAL path in place (HGCAL entrance layers, onHGCal dispatch, and the
# null-geometry guards in Calorimeter/CaloGeometryHelper) it can stay on.
(phase2_fastSim & ~phase2_hgcal).toModify(fastSimProducer, simulateCalorimetry = False)
