import FWCore.ParameterSet.Config as cms

# Copied here so python will auto-translate the names
# Now beta function vertex smearing 
from FastSimulation.Event.Early10TeVCollisionVertexGenerator_cfi import *
# Gaussian or flat or no primary vertex smearing
# include "FastSimulation/Event/data/GaussianVertexGenerator.cfi"
# include "FastSimulation/Event/data/FlatVertexGenerator.cfi"
# include "FastSimulation/Event/data/NoVertexGenerator.cfi"

famosPileUp = cms.EDProducer("SequentialPileUpProducer",
    PileUpSimulator = cms.PSet(
        fileNames = cms.untracked.vstring(
            'MinBias_001.root', 
            'MinBias_002.root', 
            'MinBias_003.root', 
            'MinBias_004.root', 
            'MinBias_005.root',
            'MinBias_006.root', 
            'MinBias_007.root', 
            'MinBias_008.root', 
            'MinBias_009.root', 
            'MinBias_010.root'
        ),
        averageNumber = cms.double(0.0),
        startingEvent = cms.untracked.uint32(0),
        jobNumber = cms.untracked.uint32(0),
        nEventsPerJob = cms.untracked.uint32(0)
    ),
    VertexGenerator = cms.PSet(myVertexGenerator)
)
