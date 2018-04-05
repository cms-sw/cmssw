import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights

# cfi from HGCalElectronIDValueMapProducer::fillDescriptions()
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cfi import *
hgcalElectronIDValueMap.dEdXWeights = dEdX_weights
