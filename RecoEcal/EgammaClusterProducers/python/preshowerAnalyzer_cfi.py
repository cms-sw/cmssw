import FWCore.ParameterSet.Config as cms

#
#  simple analyzer to make histos within a framework job off the super clusters in the event
#  Author: Shahram Rahatlou, University of Rome & INFN
#  $Id: preshowerAnalyzer.cfi,v 1.5 2007/02/14 15:37:41 futyand Exp $
#
preshowerAnalyzer = cms.EDAnalyzer("PreshowerAnalyzer",
    islandEndcapSuperClusterProducer2 = cms.string('correctedEndcapSuperClustersWithPreshower'),
    preshCalibGamma = cms.double(0.024), ## 0.020

    outputFile = cms.string('preshowerAnalyzer.root'),
    islandEndcapSuperClusterCollection2 = cms.string(''),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    nBinSC = cms.int32(60),
    EmaxDE = cms.double(50.0),
    islandEndcapSuperClusterCollection1 = cms.string(''),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05), ## 78.5e-6 

    # building endcap association
    islandEndcapSuperClusterProducer1 = cms.string('correctedIslandEndcapSuperClusters'),
    EmaxSC = cms.double(300.0),
    EminDE = cms.double(0.0),
    nBinDE = cms.int32(25),
    EminSC = cms.double(0.0),
    preshClusterProducer = cms.string('correctedEndcapSuperClustersWithPreshower')
)


