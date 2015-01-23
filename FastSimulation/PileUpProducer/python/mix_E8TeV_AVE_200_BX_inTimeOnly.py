import FWCore.ParameterSet.Config as cms
#from FastSimulation.PileUpProducer.PileUpSimulator8TeV_cfi import *

from FastSimulation.Configuration.CommonInputs_cff import *

from FastSimulation.Configuration.MixingHitsAndTracks_cff import *
mixSimCaloHits.input.nbPileupEvents.averageNumber = cms.double(200.0) 
mixSimCaloHits.input.type = cms.string('poisson')
mixRecoTracks.input.nbPileupEvents.averageNumber = cms.double(200.0) 
mixRecoTracks.input.type = cms.string('poisson')
