import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX

# cfi from HGCalPhotonIDValueMapProducer::fillDescriptions()
from RecoEgamma.EgammaTools.hgcalPhotonIDValueMap_cfi import *
hgcalPhotonIDValueMap.dEdXWeights = dEdX.weights
