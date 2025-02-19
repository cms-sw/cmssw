import FWCore.ParameterSet.Config as cms

#Sequence for making HFEMClusters into RecoEcalCandidates
#
#create HFEMClusterShapes and SuperCluster
from RecoEgamma.EgammaHFProducers.hfClusterShapes_cfi import *
#create RecoEcalCandidates
from RecoEgamma.EgammaHFProducers.hfRecoEcalCandidate_cfi import *
hfEMClusteringSequence = cms.Sequence(hfEMClusters+hfRecoEcalCandidate)

