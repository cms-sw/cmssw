import FWCore.ParameterSet.Config as cms


from RecoEgamma.EgammaIsolationAlgos.particleBasedIsoProducer_cfi import *

particleBasedIsolationSequence = cms.Sequence(particleBasedIsolation) 
