import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *

ootPhotons = photons.clone()
ootPhotons.photonCoreProducer = cms.InputTag('ootPhotonCore')
