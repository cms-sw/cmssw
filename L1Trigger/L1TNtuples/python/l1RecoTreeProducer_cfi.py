import FWCore.ParameterSet.Config as cms

l1RecoTreeProducer = cms.EDAnalyzer("L1RecoTreeProducer",

  jetTag                  = cms.untracked.InputTag("ak5CaloJets"),
  jetIdTag                = cms.untracked.InputTag("ak5JetID"),
  metTag                  = cms.untracked.InputTag("met"),
  ebRecHitsTag            = cms.untracked.InputTag("ecalRecHit:EcalRecHitsEB"),
  eeRecHitsTag            = cms.untracked.InputTag("ecalRecHit:EcalRecHitsEE"),
  superClustersBarrelTag  = cms.untracked.InputTag("hybridSuperClusters"),
  superClustersEndcapTag  = cms.untracked.InputTag("multi5x5SuperClusters:multi5x5EndcapSuperClusters"),
  basicClustersBarrelTag  = cms.untracked.InputTag("multi5x5BasicClusters:multi5x5BarrelBasicClusters"),
  basicClustersEndcapTag  = cms.untracked.InputTag("multi5x5BasicClusters:multi5x5EndcapBasicClusters"),
  verticesTag             = cms.untracked.InputTag("offlinePrimaryVertices"),
  tracksTag               = cms.untracked.InputTag("generalTracks"),
  jetCorrectorServiceName = cms.untracked.string("ak5CaloL1FastL2L3"),
  #jetCorrectorServiceName = cms.untracked.string("ak5CaloL1L2L3Residual"),

  maxCl  = cms.uint32(20),
  maxVtx = cms.uint32(10),
  maxJet = cms.uint32(20),
  maxTrk = cms.uint32(100),

  jetptThreshold = cms.double(10)
)

