import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFClusterJets_cfi import ak4PFClusterJets
from RecoJets.JetProducers.PFClustersForJets_cff import *



recoPFClusterJets   =cms.Sequence(pfClusterRefsForJetsHCAL+pfClusterRefsForJetsECAL+pfClusterRefsForJets+ak4PFClusterJets)

recoAllPFClusterJets=cms.Sequence(pfClusterRefsForJetsHCAL+pfClusterRefsForJetsECAL+pfClusterRefsForJets+ak4PFClusterJets)
