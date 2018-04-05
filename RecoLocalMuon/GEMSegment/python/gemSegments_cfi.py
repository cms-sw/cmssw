import FWCore.ParameterSet.Config as cms

gemSegments = cms.EDProducer("GEMSegmentProducer",
    gemRecHitLabel = cms.InputTag("gemRecHits"),
    algo_name = cms.string("GEMSegmentAlgorithm"),
    algo_pset = cms.PSet(
        GEMDebug = cms.untracked.bool(True),
        minHitsPerSegment = cms.uint32(2),
        preClustering = cms.bool(True),            # False => all hits in chamber are given to the fitter 
        dXclusBoxMax = cms.double(1.),             # Clstr Hit dPhi
        dYclusBoxMax = cms.double(5.),             # Clstr Hit dEta
        preClusteringUseChaining = cms.bool(True), # True ==> use Chaining() , False ==> use Clustering() Fnct
        dPhiChainBoxMax = cms.double(.02),         # Chain Hit dPhi
        dEtaChainBoxMax = cms.double(.05),         # Chain Hit dEta
        maxRecHitsInCluster = cms.int32(4),        # Does 4 make sense here?
        clusterOnlySameBXRecHits = cms.bool(True), # only working for (preClustering && preClusteringUseChaining)
    ),
)
