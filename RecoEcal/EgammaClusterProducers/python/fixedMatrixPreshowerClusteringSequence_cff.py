import FWCore.ParameterSet.Config as cms

#
# $Id: fixedMatrixPreshowerClusteringSequence.cff,v 1.1 2008/03/03 21:46:00 dlevans Exp $
#
#------------------
#Preshower clustering:
#------------------
# producer for endcap SuperClusters including preshower energy
from RecoEcal.EgammaClusterProducers.correctedFixedMatrixSuperClustersWithPreshower_cfi import *
# producer for preshower cluster shapes
from RecoEcal.EgammaClusterProducers.fixedMatrixPreshowerClusterShape_cfi import *
# create sequence for preshower clustering
fixedMatrixPreshowerClusteringSequence = cms.Sequence(correctedFixedMatrixSuperClustersWithPreshower*fixedMatrixPreshowerClusterShape)

