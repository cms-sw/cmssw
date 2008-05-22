import FWCore.ParameterSet.Config as cms

#
# $Id: multi5x5PreshowerClusteringSequence.cff,v 1.1 2008/04/29 15:27:38 dlevans Exp $
#
#------------------
#Preshower clustering:
#------------------
# producer for endcap SuperClusters including preshower energy
from RecoEcal.EgammaClusterProducers.correctedMulti5x5SuperClustersWithPreshower_cfi import *
# producer for preshower cluster shapes
from RecoEcal.EgammaClusterProducers.multi5x5PreshowerClusterShape_cfi import *
# create sequence for preshower clustering
multi5x5PreshowerClusteringSequence = cms.Sequence(correctedMulti5x5SuperClustersWithPreshower*multi5x5PreshowerClusterShape)

