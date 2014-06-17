import FWCore.ParameterSet.Config as cms


from RecoEgamma.EgammaAlgos.particleBasedIsoProducer_cfi import *

particleBasedIsolationSequence = cms.Sequence(particleBasedIsolation) 
