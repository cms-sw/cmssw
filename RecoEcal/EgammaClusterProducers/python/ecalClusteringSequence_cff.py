import FWCore.ParameterSet.Config as cms

#
# $Id: ecalClusteringSequence.cff,v 1.4 2008/04/29 15:51:48 dlevans Exp $
# complete sequence of clustering in ecal barrel & endcap + preshower
# Shahram Rahatlou, University of Rome & INFN, 3 Aug 2006
#
# geometry needed for clustering
from RecoEcal.EgammaClusterProducers.geometryForClustering_cff import *
# create sequence for island clustering
from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *
# hybrid clustering sequence
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
# clusters in preshower
from RecoEcal.EgammaClusterProducers.preshowerClusteringSequence_cff import *
# dynamic hybrid sequence
from RecoEcal.EgammaClusterProducers.dynamicHybridClusteringSequence_cff import *
# multi5x5 clusters
from RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff import *
# preshower sequence for multi5x5 clusters
from RecoEcal.EgammaClusterProducers.multi5x5PreshowerClusteringSequence_cff import *
# create path with all clustering algos
# NB: preshower MUST be run after island clustering in the endcap
ecalClusteringSequence = cms.Sequence(islandClusteringSequence*hybridClusteringSequence*preshowerClusteringSequence*dynamicHybridClusteringSequence*multi5x5ClusteringSequence*multi5x5PreshowerClusteringSequence)

