import FWCore.ParameterSet.Config as cms

# Here so that python translator can see the names
# Now beta function vertex smearing 
from FastSimulation.Event.EarlyCollisionVertexGenerator_cfi import *
from FastSimulation.Event.ParticleFilter_cfi import *
from FastSimulation.MaterialEffects.MaterialEffects_cfi import *
from FastSimulation.TrajectoryManager.ActivateDecays_cfi import *
from FastSimulation.TrajectoryManager.TrackerSimHits_cfi import *
from FastSimulation.Calorimetry.Calorimetry_cff import *
famosSimHits = cms.EDProducer("FamosProducer",
    # FastCalorimetry
    FamosCalorimetryBlock,
    # Conditions to save Tracker SimHits 
    TrackerSimHitsBlock,
    # Material effects to be simulated in the tracker material and associated cuts
    MaterialEffectsBlock,
    # (De)activate decays of unstable particles (K0S, etc...)
    ActivateDecaysBlock,
    # include "FastSimulation/Event/data/NominalCollisionVertexGenerator.cfi"
    # include "FastSimulation/Event/data/NominalCollision1VertexGenerator.cfi"
    # include "FastSimulation/Event/data/NominalCollision2VertexGenerator.cfi"
    # include "FastSimulation/Event/data/NominalCollision3VertexGenerator.cfi"
    # include "FastSimulation/Event/data/NominalCollision4VertexGenerator.cfi"
    # include "FastSimulation/Event/data/GaussianVertexGenerator.cfi"
    # include "FastSimulation/Event/data/FlatVertexGenerator.cfi"
    # include "FastSimulation/Event/data/NoVertexGenerator.cfi"
    # Kinematic cuts for the particle filter in the SimEvent
    ParticleFilterBlock,
    SimulateCalorimetry = cms.bool(True),
    SimulateMuons = cms.bool(True),
    RunNumber = cms.untracked.int32(1001),
    Verbosity = cms.untracked.int32(0),
    UseMagneticField = cms.bool(True),
    UseTRandomEngine = cms.bool(True),
    SimulateTracking = cms.bool(True),
    ApplyAlignment = cms.bool(False),
    VertexGenerator = cms.PSet(
        myVertexGenerator
    )
)


