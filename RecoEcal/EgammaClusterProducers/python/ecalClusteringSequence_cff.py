import FWCore.ParameterSet.Config as cms

#
# $Id: ecalClusteringSequence.cff,v 1.3 2008/03/03 21:46:02 dlevans Exp $
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
# fixed matrix clusters
from RecoEcal.EgammaClusterProducers.fixedMatrixClusteringSequence_cff import *
# preshower sequence for fixed matrix clusters
from RecoEcal.EgammaClusterProducers.fixedMatrixPreshowerClusteringSequence_cff import *
# create path with all clustering algos
# NB: preshower MUST be run after island clustering in the endcap
ecalClusteringSequence = cms.Sequence(islandClusteringSequence*hybridClusteringSequence*preshowerClusteringSequence*dynamicHybridClusteringSequence*fixedMatrixClusteringSequence*fixedMatrixPreshowerClusteringSequence)

