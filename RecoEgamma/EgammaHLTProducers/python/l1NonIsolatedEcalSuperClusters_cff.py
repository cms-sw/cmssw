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
# include "RecoEgamma/EgammaHLTProducers/data/hltIslandBasicClusters.cfi"
#--------- ENDCAP 
hltIslandBasicClustersEndcapL1NonIsolated = copy.deepcopy(hltIslandBasicClusters)
import copy
from RecoEgamma.EgammaHLTProducers.hltIslandBasicClusters_cfi import *
#--------- BARREL 
hltIslandBasicClustersBarrelL1NonIsolated = copy.deepcopy(hltIslandBasicClusters)
import copy
from RecoEgamma.EgammaHLTProducers.hltIslandSuperClusters_cfi import *
# SUPER CLUSTERS
#include "RecoEgamma/EgammaHLTProducers/data/hltIslandSuperClusters.cfi"
hltIslandSuperClustersL1NonIsolated = copy.deepcopy(hltIslandSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedIslandEndcapSuperClusters_cfi import *
# Energy scale correction for Island SuperClusters
#include "RecoEcal/EgammaClusterProducers/data/correctedIslandSuperClusters.cfi"
#--------- ENDCAP 
correctedIslandEndcapSuperClustersL1NonIsolated = copy.deepcopy(correctedIslandEndcapSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedIslandBarrelSuperClusters_cfi import *
#--------- BARREL
correctedIslandBarrelSuperClustersL1NonIsolated = copy.deepcopy(correctedIslandBarrelSuperClusters)
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
hltHybridSuperClustersL1NonIsolated = copy.deepcopy(hltHybridSuperClusters)
import copy
from RecoEcal.EgammaClusterProducers.correctedHybridSuperClusters_cfi import *
# Energy scale correction for Hybrid SuperClusters
#include "RecoEcal/EgammaClusterProducers/data/correctedHybridSuperClusters.cfi"
correctedHybridSuperClustersL1NonIsolated = copy.deepcopy(correctedHybridSuperClusters)
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
correctedEndcapSuperClustersWithPreshowerL1NonIsolated = copy.deepcopy(correctedEndcapSuperClustersWithPreshower)
l1NonIsolatedEcalClusters = cms.Sequence(hltIslandBasicClustersEndcapL1NonIsolated*hltIslandBasicClustersBarrelL1NonIsolated*hltHybridSuperClustersL1NonIsolated*hltIslandSuperClustersL1NonIsolated*correctedIslandEndcapSuperClustersL1NonIsolated*correctedIslandBarrelSuperClustersL1NonIsolated*correctedHybridSuperClustersL1NonIsolated*correctedEndcapSuperClustersWithPreshowerL1NonIsolated)
hltIslandBasicClustersEndcapL1NonIsolated.l1LowerThr = 5.
hltIslandBasicClustersEndcapL1NonIsolated.doBarrel = False
hltIslandBasicClustersEndcapL1NonIsolated.doEndcaps = True
hltIslandBasicClustersEndcapL1NonIsolated.doIsolated = False
hltIslandBasicClustersBarrelL1NonIsolated.l1LowerThr = 5.
hltIslandBasicClustersBarrelL1NonIsolated.doBarrel = True
hltIslandBasicClustersBarrelL1NonIsolated.doEndcaps = False
hltIslandBasicClustersBarrelL1NonIsolated.doIsolated = False
hltIslandSuperClustersL1NonIsolated.endcapClusterProducer = 'hltIslandBasicClustersEndcapL1NonIsolated'
hltIslandSuperClustersL1NonIsolated.barrelClusterProducer = 'hltIslandBasicClustersBarrelL1NonIsolated'
hltIslandSuperClustersL1NonIsolated.doBarrel = True
correctedIslandEndcapSuperClustersL1NonIsolated.rawSuperClusterProducer = 'hltIslandSuperClustersL1NonIsolated'
correctedIslandBarrelSuperClustersL1NonIsolated.rawSuperClusterProducer = 'hltIslandSuperClustersL1NonIsolated'
hltHybridSuperClustersL1NonIsolated.l1LowerThr = 5.
hltHybridSuperClustersL1NonIsolated.HybridBarrelSeedThr = 1.5
hltHybridSuperClustersL1NonIsolated.doIsolated = False
correctedHybridSuperClustersL1NonIsolated.rawSuperClusterProducer = 'hltHybridSuperClustersL1NonIsolated'
correctedHybridSuperClustersL1NonIsolated.etThresh = 5.0
correctedEndcapSuperClustersWithPreshowerL1NonIsolated.endcapSClusterProducer = 'correctedIslandEndcapSuperClustersL1NonIsolated'
correctedEndcapSuperClustersWithPreshowerL1NonIsolated.etThresh = 5.0
# create path with all clustering algos
# NB: preshower MUST be run after island clustering in the endcap
#sequence ecalClusters = {islandBasicClusters, 
#                          islandClusteringSequence,
#                          hybridClusteringSequence,
#                          preshowerClusteringSequence
#                        }
#    string barrelHitProducer   = "ecalRegionalEgammaRecHit"
#    string endcapHitProducer   = "ecalRegionalEgammaRecHit"
hltIslandBasicClustersEndcapL1NonIsolated.endcapHitProducer = 'ecalRegionalEgammaRecHit'
hltIslandBasicClustersBarrelL1NonIsolated.barrelHitProducer = 'ecalRegionalEgammaRecHit'
correctedIslandEndcapSuperClustersL1NonIsolated.recHitProducer = 'ecalRegionalEgammaRecHit'
correctedIslandBarrelSuperClustersL1NonIsolated.recHitProducer = 'ecalRegionalEgammaRecHit'
hltHybridSuperClustersL1NonIsolated.ecalhitproducer = 'ecalRegionalEgammaRecHit'
correctedHybridSuperClustersL1NonIsolated.recHitProducer = 'ecalRegionalEgammaRecHit'

