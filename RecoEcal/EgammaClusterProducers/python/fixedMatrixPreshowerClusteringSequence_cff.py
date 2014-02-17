import FWCore.ParameterSet.Config as cms

#
# $Id: fixedMatrixPreshowerClusteringSequence_cff.py,v 1.2 2008/04/21 03:24:05 rpw Exp $
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

