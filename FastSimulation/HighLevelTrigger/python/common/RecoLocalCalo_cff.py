import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalPreshowerRecHit = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalRecHit = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
hbhereco = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
horeco = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
hfreco = copy.deepcopy(caloRecHitCopy)
ecalLocalRecoSequence = cms.Sequence(ecalRecHit+ecalPreshowerRecHit)
ecalLocalRecoSequence_nopreshower = cms.Sequence(ecalRecHit)
hcalLocalRecoSequence = cms.Sequence(hbhereco+hfreco+horeco)
ecalPreshowerRecHit.OutputRecHitCollections = ['EcalRecHitsES']
ecalPreshowerRecHit.InputRecHitCollectionTypes = [1]
ecalPreshowerRecHit.InputRecHitCollections = cms.VInputTag(cms.InputTag("caloRecHits","EcalRecHitsES"))
hbhereco.OutputRecHitCollections = ['none']
hbhereco.InputRecHitCollectionTypes = [4]
hbhereco.InputRecHitCollections = ['caloRecHits']
horeco.OutputRecHitCollections = ['none']
horeco.InputRecHitCollectionTypes = [5]
horeco.InputRecHitCollections = ['caloRecHits']
hfreco.OutputRecHitCollections = ['none']
hfreco.InputRecHitCollectionTypes = [6]
hfreco.InputRecHitCollections = ['caloRecHits']

