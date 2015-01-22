import FWCore.ParameterSet.Config as cms

from FastSimulation.CaloRecHitsProducer.FullDigisPlusRecHits_cff import *
caloDigis = cms.Sequence(DigiSequence)
caloRecHits = cms.Sequence(ecalRecHitSequence*hcalRecHitSequence)
    

