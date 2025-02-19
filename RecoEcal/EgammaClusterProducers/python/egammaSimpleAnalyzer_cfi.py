import FWCore.ParameterSet.Config as cms

#
#  simple analyzer to make histos within a framework job off the super clusters in the event
#  Author: Shahram Rahatlou, University of Rome & INFN
#  $Id: egammaSimpleAnalyzer_cfi.py,v 1.2 2008/04/21 03:24:02 rpw Exp $
#
egammaSimpleAnalyzer = cms.EDAnalyzer("EgammaSimpleAnalyzer",
    xMaxHist = cms.double(60.0),
    outputFile = cms.string('egammaAnalyzer.root'),
    #
    # island clustering in endcap
    #
    islandEndcapBasicClusterProducer = cms.string('islandBasicClusters'),
    islandEndcapSuperClusterCollection = cms.string('islandEndcapSuperClusters'),
    islandBarrelBasicClusterShapes = cms.string('islandBarrelShape'),
    correctedHybridSuperClusterProducer = cms.string('correctedHybridSuperClusters'),
    islandEndcapBasicClusterCollection = cms.string('islandEndcapBasicClusters'),
    correctedIslandEndcapSuperClusterProducer = cms.string('correctedEndcapSuperClustersWithPreshower'),
    hybridSuperClusterCollection = cms.string(''),
    xMinHist = cms.double(0.0),
    islandEndcapSuperClusterProducer = cms.string('islandSuperClusters'),
    nbinHist = cms.int32(200),
    correctedHybridSuperClusterCollection = cms.string(''),
    #
    # island clustering in barrel
    #
    islandBarrelBasicClusterProducer = cms.string('islandBasicClusters'),
    islandEndcapBasicClusterShapes = cms.string('islandEndcapShape'),
    #
    # hybrid clustering in barrel
    #
    hybridSuperClusterProducer = cms.string('hybridSuperClusters'),
    islandBarrelBasicClusterCollection = cms.string('islandBarrelBasicClusters'),
    correctedIslandEndcapSuperClusterCollection = cms.string('')
)


