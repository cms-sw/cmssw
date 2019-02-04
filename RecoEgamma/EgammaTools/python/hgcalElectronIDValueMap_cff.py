import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX

# cfi from HGCalElectronIDValueMapProducer::fillDescriptions()
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cfi import *
hgcalElectronIDValueMap.dEdXWeights = dEdX.weights
