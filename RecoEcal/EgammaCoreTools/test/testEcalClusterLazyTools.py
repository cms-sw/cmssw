from FWCore.ParameterSet.Config import *

process = Process("test")
process.extend(include("FWCore/MessageLogger/data/MessageLogger.cfi"))
#process.MessageLogger.cerr.FwkReport.reportEvery = 50

process.extend(include("RecoEcal/EgammaClusterProducers/data/geometryForClustering.cff"))

input_files = vstring()
#input_files.append( "/store/relval/2008/4/17/RelVal-RelValTTbar-1208465820/0000/0ABDA540-EE0C-DD11-BA9F-000423D94990.root" )
input_files.append( "file:/data/ferriff/ClusterMulti5x5/CMSSW_2_1_X_2008-05-14-0200/src/reco.root" )

process.source = Source("PoolSource",
    fileNames = untracked( input_files )
)

process.maxEvents = untracked.PSet( input = untracked.int32( 10 ) )


process.testEcalClusterLazyTools = EDAnalyzer("testEcalClusterLazyTools",
    reducedBarrelRecHitCollection = InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = InputTag("reducedEcalRecHitsEE"),
    barrelClusterCollection = InputTag("hybridSuperClusters:"),
    endcapClusterCollection = InputTag("multi5x5BasicClusters:multi5x5EndcapBasicClusters")
)

process.p1 = Path( process.testEcalClusterLazyTools )
