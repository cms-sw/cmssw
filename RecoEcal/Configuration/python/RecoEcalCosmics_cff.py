import FWCore.ParameterSet.Config as cms

# Sequence for clustering in ecal barrel & endcap + preshower
# create sequence for island clustering
#include "RecoEcal/EgammaClusterProducers/data/islandClusteringSequence.cff"
# hybrid clustering sequence
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
# clusters in preshower
#include "RecoEcal/EgammaClusterProducers/data/preshowerClusteringSequence.cff"
# dynamic hybrid sequence
#include "RecoEcal/EgammaClusterProducers/data/dynamicHybridClusteringSequence.cff"
# fixed matrix clusters
#include "RecoEcal/EgammaClusterProducers/data/fixedMatrixClusteringSequence.cff"
# preshower sequence for fixed matrix clusters
#include "RecoEcal/EgammaClusterProducers/data/fixedMatrixPreshowerClusteringSequence.cff"
# cosmic bump finder
from RecoEcal.EgammaClusterProducers.cosmicClusteringSequence_cff import *
# create path with all clustering algos
# NB: preshower MUST be run after island clustering in the endcap
ecalClusters = cms.Sequence(hybridClusteringSequence*cosmicClusteringSequence)

