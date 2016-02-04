import FWCore.ParameterSet.Config as cms

#
# $Id: preshowerClusteringSequence_cff.py,v 1.2 2008/04/21 03:24:16 rpw Exp $
#
#------------------
#Preshower clustering:
#------------------
# producer for endcap SuperClusters including preshower energy
from RecoEcal.EgammaClusterProducers.correctedEndcapSuperClustersWithPreshower_cfi import *
# producer for preshower cluster shapes
from RecoEcal.EgammaClusterProducers.preshowerClusterShape_cfi import *
# create sequence for preshower clustering
preshowerClusteringSequence = cms.Sequence(correctedEndcapSuperClustersWithPreshower*preshowerClusterShape)

