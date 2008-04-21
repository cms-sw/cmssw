import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalRegionalEgammaRecHit = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalRegionalMuonsRecHit = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalRegionalTausRecHit = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalRegionalJetsRecHit = copy.deepcopy(caloRecHitCopy)
import copy
from FastSimulation.CaloRecHitsProducer.CaloRecHitCopy_cfi import *
ecalRecHitAll = copy.deepcopy(caloRecHitCopy)
ecalRegionalEgammaRecoSequence = cms.Sequence(ecalRegionalEgammaRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalRegionalMuonsRecoSequence = cms.Sequence(ecalRegionalMuonsRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalRegionalTausRecoSequence = cms.Sequence(ecalRegionalTausRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalRegionalJetsRecoSequence = cms.Sequence(ecalRegionalJetsRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalAllRecoSequence = cms.Sequence(ecalRecHitAll+cms.SequencePlaceholder("ecalPreshowerRecHit"))

