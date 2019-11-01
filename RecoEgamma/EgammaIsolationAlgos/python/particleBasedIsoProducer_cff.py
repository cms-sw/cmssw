import FWCore.ParameterSet.Config as cms


from RecoEgamma.EgammaIsolationAlgos.particleBasedIsoProducer_cfi import *

particleBasedIsolationTask = cms.Task(particleBasedIsolation) 
particleBasedIsolationSequence = cms.Sequence(particleBasedIsolationTask) 
