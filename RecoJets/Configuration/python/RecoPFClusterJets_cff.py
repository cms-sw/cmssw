import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak5PFClusterJets_cfi import ak5PFClusterJets
from RecoJets.JetProducers.PFClustersForJets_cff import *



recoPFClusterJets   =cms.Sequence(pfClusterRefsForJetsHCAL+pfClusterRefsForJetsECAL+pfClusterRefsForJets+ak5PFClusterJets)

recoAllPFClusterJets=cms.Sequence(pfClusterRefsForJetsHCAL+pfClusterRefsForJetsECAL+pfClusterRefsForJets+ak5PFClusterJets)
