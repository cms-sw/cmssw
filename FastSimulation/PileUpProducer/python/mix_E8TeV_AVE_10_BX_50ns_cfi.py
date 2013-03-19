from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *

from FastSimulation.Configuration.CommonInputs_cff import *

if (MixingMode==2):
    # mix at SIM level:
    from FastSimulation.Configuration.MixingHitsAndTracks_cff import *
    mixSimCaloHits.input.nbPileupEvents.averageNumber = cms.double(10.0) 
    mixSimCaloHits.input.type = cms.string('poisson')
    mixRecoTracks.input.nbPileupEvents.averageNumber = cms.double(10.0) 
    mixRecoTracks.input.type = cms.string('poisson')
else:
    # mix at GEN level:
    from FastSimulation.Configuration.MixingFull_cff import *
    mixGenPU.input.nbPileupEvents.averageNumber = cms.double(10.0) 
    mixGenPU.input.type = cms.string('poisson')

