import FWCore.ParameterSet.Config as cms

# Copied here so python will auto-translate the names
# Now beta function vertex smearing 
from FastSimulation.Event.EarlyCollisionVertexGenerator_cfi import *
from FastSimulation.PileUpProducer.PileUpSimulator_cfi import *
# Gaussian or flat or no primary vertex smearing
# include "FastSimulation/Event/data/GaussianVertexGenerator.cfi"
# include "FastSimulation/Event/data/FlatVertexGenerator.cfi"
# include "FastSimulation/Event/data/NoVertexGenerator.cfi"
famosPileUp = cms.EDProducer("PileUpProducer",
    # The conditions for pile-up event generation
    PileUpSimulatorBlock,
    UseTRandomEngine = cms.bool(True),
    VertexGenerator = cms.PSet(
        myVertexGenerator
    )
)


