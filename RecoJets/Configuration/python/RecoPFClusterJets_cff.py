import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak5PFClusterJets_cfi import ak5PFClusterJets
from RecoJets.JetProducers.PFClustersForJets_cff import *

recoPFClusterJetsTask   =cms.Task(pfClusterRefsForJetsHCAL,
                                  pfClusterRefsForJetsECAL,
                                  pfClusterRefsForJets,
                                  ak5PFClusterJets)
recoPFClusterJets   =cms.Sequence(recoPFClusterJetsTask)
recoAllPFClusterJets=cms.Sequence(recoPFClusterJetsTask)
