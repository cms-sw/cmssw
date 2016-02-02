import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonProducer_cfi import *

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets

#compute areas for Fastjet PU subtraction  
kt4PFJets.doRhoFastjet = True
kt4PFJets.doAreaFastjet = True
#use active areas and not Voronoi tessellation for the moment
kt4PFJets.voronoiRfact = 0.9

#change input tag
pfInput = cms.InputTag('particleFlowTmp')
kt4PFJets.src = pfInput

#from HiJetBackground.HiFJRhoProducer.hiFJRhoProducer import hiFJRhoProducer
from HiJetBackground.HiFJRhoProducer.hiFJRhoAnalyzer import hiFJRhoAnalyzer

hiFJBkg=cms.Sequence(kt4PFJets+
                           #                           hiFJRhoProducer
                           hiFJRhoAnalyzer
                           )
