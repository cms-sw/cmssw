import FWCore.ParameterSet.Config as cms

#include "RecoEgamma/EgammaHLTProducers/data/hltIslandBasicClusters.cfi"
#module  l1IsoIslandBasicClusters= hltIslandBasicClusters from "RecoEgamma/EgammaHLTProducers/data/hltIslandBasicClusters.cfi"
#hltHybridSuperClusters.cfi                    
#hltIslandSuperClusters.cfi
# Sequence for clustering in ecal barrel & endcap + preshower
# geometry needed for clustering
from RecoEcal.EgammaClusterProducers.geometryForClustering_cff import *
import copy
from RecoEgamma.EgammaHLTProducers.hltIslandBasicClusters_cfi import *
# island clustering sequence
# BASIC CLUSTERS
#MARCO include "RecoEgamma/EgammaHLTProducers/data/hltIslandBasicClusters.cfi"
#--------- ENDCAP 
hltIslandBasicClustersEndcapL1Isolated = copy.deepcopy(hltIslandBasicClusters)
import copy
from RecoEgamma.EgammaHLTProducers.hltIslandBasicClusters_cfi import *
#--------- BARREL 
hltIslandBasicClustersBarrelL1Isolated = copy.deepcopy(hltIslandBasicClusters)
import copy
from RecoEgamma.EgammaHLTProducers.hltIslandSuperClusters_cfi import *
# SUPER CLUSTERS
#include "RecoEgamma/EgammaHLTProducers/data/hltIslandSuperClusters.cfi"
hltIslandSuperClustersL1Isolated = copy.deepcopy(hltIslandSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedIslandEndcapSuperClusters_cfi import *
# Energy scale correction for Island SuperClusters
#include "RecoEcal/EgammaClusterProducers/data/correctedIslandSuperClusters.cfi"
#--------- ENDCAP 
correctedIslandEndcapSuperClustersL1Isolated = copy.deepcopy(correctedIslandEndcapSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedIslandBarrelSuperClusters_cfi import *
#--------- BARREL
correctedIslandBarrelSuperClustersL1Isolated = copy.deepcopy(correctedIslandBarrelSuperClusters)
import copy
from RecoEgamma.EgammaHLTProducers.hltHybridSuperClusters_cfi import *
# create sequence for island clustering
#sequence islandClusteringSequence = {
#             islandBasicClusters,
#             islandSuperClusters,
#             correctedIslandBarrelSuperClusters,
#             correctedIslandEndcapSuperClusters
#}
# hybrid clustering sequence
#include "RecoEgamma/EgammaHLTProducers/data/hltHybridSuperClusters.cfi"
hltHybridSuperClustersL1Isolated = copy.deepcopy(hltHybridSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedHybridSuperClusters_cfi import *
# Energy scale correction for Hybrid SuperClusters
#include "RecoEcal/EgammaClusterProducers/data/correctedHybridSuperClusters.cfi"
correctedHybridSuperClustersL1Isolated = copy.deepcopy(correctedHybridSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedEndcapSuperClustersWithPreshower_cfi import *
# hybrid clustering sequence
#sequence hybridClusteringSequence = {
#           hybridSuperClusters,
#           correctedHybridSuperClusters }
# HERE HERE
# preshower clustering sequence
#include "RecoEcal/EgammaClusterProducers/data/preshowerClusteringSequence.cff"
#include "RecoEcal/EgammaClusterProducers/data/correctedEndcapSuperClustersWithPreshower.cfi"
correctedEndcapSuperClustersWithPreshowerL1Isolated = copy.deepcopy(correctedEndcapSuperClustersWithPreshower)
l1IsolatedEcalClusters = cms.Sequence(hltIslandBasicClustersEndcapL1Isolated*hltIslandBasicClustersBarrelL1Isolated*hltHybridSuperClustersL1Isolated*hltIslandSuperClustersL1Isolated*correctedIslandEndcapSuperClustersL1Isolated*correctedIslandBarrelSuperClustersL1Isolated*correctedHybridSuperClustersL1Isolated*correctedEndcapSuperClustersWithPreshowerL1Isolated)
hltIslandBasicClustersEndcapL1Isolated.l1LowerThr = 5.
hltIslandBasicClustersEndcapL1Isolated.doBarrel = False
hltIslandBasicClustersEndcapL1Isolated.doEndcaps = True
hltIslandBasicClustersEndcapL1Isolated.doIsolated = True
hltIslandBasicClustersBarrelL1Isolated.l1LowerThr = 5.
hltIslandBasicClustersBarrelL1Isolated.doBarrel = True
hltIslandBasicClustersBarrelL1Isolated.doEndcaps = False
hltIslandBasicClustersBarrelL1Isolated.doIsolated = True
hltIslandSuperClustersL1Isolated.endcapClusterProducer = 'hltIslandBasicClustersEndcapL1Isolated'
hltIslandSuperClustersL1Isolated.barrelClusterProducer = 'hltIslandBasicClustersBarrelL1Isolated'
hltIslandSuperClustersL1Isolated.doBarrel = True
correctedIslandEndcapSuperClustersL1Isolated.rawSuperClusterProducer = 'hltIslandSuperClustersL1Isolated'
correctedIslandBarrelSuperClustersL1Isolated.rawSuperClusterProducer = 'hltIslandSuperClustersL1Isolated'
hltHybridSuperClustersL1Isolated.l1LowerThr = 5.
hltHybridSuperClustersL1Isolated.HybridBarrelSeedThr = 1.5
correctedHybridSuperClustersL1Isolated.rawSuperClusterProducer = 'hltHybridSuperClustersL1Isolated'
correctedHybridSuperClustersL1Isolated.etThresh = 5.0
correctedEndcapSuperClustersWithPreshowerL1Isolated.endcapSClusterProducer = 'correctedIslandEndcapSuperClustersL1Isolated'
correctedEndcapSuperClustersWithPreshowerL1Isolated.etThresh = 5.0
# create path with all clustering algos
# NB: preshower MUST be run after island clustering in the endcap
#sequence ecalClusters = {islandBasicClusters, 
#                          islandClusteringSequence,
#                          hybridClusteringSequence,
#                          preshowerClusteringSequence
#                        }
#    string barrelHitProducer   = "ecalRegionalEgammaRecHit"
#    string endcapHitProducer   = "ecalRegionalEgammaRecHit"
hltIslandBasicClustersEndcapL1Isolated.endcapHitProducer = 'ecalRegionalEgammaRecHit'
hltIslandBasicClustersBarrelL1Isolated.barrelHitProducer = 'ecalRegionalEgammaRecHit'
correctedIslandEndcapSuperClustersL1Isolated.recHitProducer = 'ecalRegionalEgammaRecHit'
correctedIslandBarrelSuperClustersL1Isolated.recHitProducer = 'ecalRegionalEgammaRecHit'
hltHybridSuperClustersL1Isolated.ecalhitproducer = 'ecalRegionalEgammaRecHit'
correctedHybridSuperClustersL1Isolated.recHitProducer = 'ecalRegionalEgammaRecHit'

