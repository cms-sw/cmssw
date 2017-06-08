import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *

mustacheOOTPhotons = photons.clone()
mustacheOOTPhotons.photonCoreProducer = cms.InputTag('mustacheOOTPhotonCore')

