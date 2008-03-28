import FWCore.ParameterSet.Config as cms

# Copied here so python will auto-translate the names
# Now beta function vertex smearing 
from FastSimulation.Event.EarlyCollisionVertexGenerator_cfi import *
# The conditions for pile-up event generation
from FastSimulation.PileUpProducer.PileUpSimulator_cfi import *
# Gaussian or flat or no primary vertex smearing
# include "FastSimulation/Event/data/GaussianVertexGenerator.cfi"
# include "FastSimulation/Event/data/FlatVertexGenerator.cfi"
# include "FastSimulation/Event/data/NoVertexGenerator.cfi"
famosPileUp = cms.EDProducer("PileUpProducer",
    UseTRandomEngine = cms.bool(True),
    PileUpSimulator = cms.PSet(
        inputFile = cms.untracked.string('PileUpInputFile.txt'),
        fileNames = cms.untracked.vstring('MinBias_001.root', 'MinBias_002.root', 'MinBias_003.root', 'MinBias_004.root', 'MinBias_005.root', 'MinBias_006.root', 'MinBias_007.root', 'MinBias_008.root', 'MinBias_009.root', 'MinBias_010.root'),
        averageNumber = cms.double(0.0)
    ),
    VertexGenerator = cms.PSet(
        myVertexGenerator
    )
)


