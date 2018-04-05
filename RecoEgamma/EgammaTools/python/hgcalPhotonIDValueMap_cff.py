import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights

# cfi from HGCalPhotonIDValueMapProducer::fillDescriptions()
from RecoEgamma.EgammaTools.hgcalPhotonIDValueMap_cfi import *
hgcalPhotonIDValueMap.dEdXWeights = dEdX_weights
