from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *

from FastSimulation.Configuration.CommonInputs_cff import *

if (MixingMode==2):
    # mix at SIM level:
    from FastSimulation.Configuration.MixingHitsAndTracks_cff import *
    mix.input.nbPileupEvents.averageNumber = cms.double(0.0) 
    mix.input.type = cms.string('poisson')
#    mixSimCaloHits.input.nbPileupEvents.averageNumber = cms.double(0.0) 
#    mixSimCaloHits.input.type = cms.string('poisson')
#    mixRecoTracks.input.nbPileupEvents.averageNumber = cms.double(0.0) 
#    mixRecoTracks.input.type = cms.string('poisson')
