from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *

from FastSimulation.Configuration.CommonInputs_cff import *

if (MixingMode=='DigiRecoMixing'):
    # mix at SIM and RECO level:
    from FastSimulation.Configuration.MixingHitsAndTracks_cff import *
    mix.input.nbPileupEvents.averageNumber = cms.double(0.0) 
    mix.input.type = cms.string('poisson')
