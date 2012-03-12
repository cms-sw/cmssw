import FWCore.ParameterSet.Config as cms

# Copied here so python will auto-translate the names
# Now beta function vertex smearing 
#from FastSimulation.Event.Early10TeVCollisionVertexGenerator_cfi import *
#from FastSimulation.Event.Realistic7TeV2011CollisionVertexGenerator_cfi import *
from FastSimulation.Event.Realistic8TeVCollisionVertexGenerator_cfi import *
# 14 TeV pile-up files
#from FastSimulation.PileUpProducer.PileUpSimulator14TeV_cfi import *
# 10 TeV pile-up files
#from FastSimulation.PileUpProducer.PileUpSimulator10TeV_cfi import *
# 7 TeV pile-up files
from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *
###
# Gaussian or flat or no primary vertex smearing
# include "FastSimulation/Event/data/GaussianVertexGenerator.cfi"
# include "FastSimulation/Event/data/FlatVertexGenerator.cfi"
# include "FastSimulation/Event/data/NoVertexGenerator.cfi"
famosPileUp = cms.EDProducer("PileUpProducer",
    # The conditions for pile-up event generation
    PileUpSimulatorBlock,
    VertexGenerator = cms.PSet(
        myVertexGenerator
    )
)


