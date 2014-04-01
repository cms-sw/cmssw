import FWCore.ParameterSet.Config as cms
#from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *

from FastSimulation.Configuration.CommonInputs_cff import *

if (MixingMode=='DigiRecoMixing'):
    # mix at SIM/RECO level:
    from FastSimulation.Configuration.MixingHitsAndTracks_cff import *
    mixSimCaloHits.input.nbPileupEvents.averageNumber = cms.double(70.0) 
    mixSimCaloHits.input.type = cms.string('poisson')
    mixRecoTracks.input.nbPileupEvents.averageNumber = cms.double(70.0) 
    mixRecoTracks.input.type = cms.string('poisson')
else:
    # mix at GEN level:
    from FastSimulation.Configuration.MixingFull_cff import *
    mixGenPU.input.nbPileupEvents.averageNumber = cms.double(70.0) 
    mixGenPU.input.type = cms.string('poisson')

