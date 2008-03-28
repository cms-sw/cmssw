import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.particleFlow_cfi import *
from RecoJets.JetProducers.kt10PFJets_cff import *
from RecoJets.JetProducers.iterativeCone5PFJets_cff import *
from RecoJets.JetProducers.midPointCone5PFJets_cff import *
recoPFJets = cms.Sequence(kt10PFJets+iterativeCone5PFJets+midPointCone5PFJets)

