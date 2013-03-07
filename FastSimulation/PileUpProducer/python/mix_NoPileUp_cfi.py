from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *

#### mix at GEN level:

from FastSimulation.Configuration.MixingFull_cff import *

mixGenPU.input.nbPileupEvents.averageNumber = cms.double(0.0) 
mixGenPU.input.type = cms.string('poisson')

# mix at SIM level:

from FastSimulation.Configuration.MixingHitsAndTracks_cff import *

mixSimCaloHits.input.nbPileupEvents.averageNumber = cms.double(0.0) 
mixSimCaloHits.input.type = cms.string('poisson')

mixSimTracksAndVertices.input.nbPileupEvents.averageNumber = cms.double(0.0) 
mixSimTracksAndVertices.input.type = cms.string('poisson')

