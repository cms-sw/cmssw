import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("test")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

input_files = cms.vstring("/store/data/Run2018A/EGamma/AOD/17Sep2018-v2/100000/01EB9686-9A6F-BF48-903A-02F7D9AEB9B9.root")

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.source = cms.Source("PoolSource", fileNames = cms.untracked( input_files ) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 10 ) )


process.testEcalClusterTools = cms.EDAnalyzer("testEcalClusterTools",
    barrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    endcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    barrelClusterCollection = cms.InputTag("hybridSuperClusters:hybridBarrelBasicClusters"),
    endcapClusterCollection = cms.InputTag("multi5x5SuperClusters:multi5x5EndcapBasicClusters")
)

process.p1 = cms.Path( process.testEcalClusterTools )
