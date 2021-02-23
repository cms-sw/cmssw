import FWCore.ParameterSet.Config as cms

gemSegments = cms.EDProducer("GEMSegmentProducer",
    gemRecHitLabel = cms.InputTag("gemRecHits"),
    ge0_name = cms.string("GE0SegAlgoRU"),
    algo_name = cms.string("GEMSegmentAlgorithm"),
    ge0_pset = cms.PSet(
        allowWideSegments = cms.bool(True),
        doCollisions = cms.bool(True),
        maxChi2Additional = cms.double(100.0),
        maxChi2Prune = cms.double(50),
        maxChi2GoodSeg = cms.double(50),
        maxPhiSeeds = cms.double(0.001096605744), #Assuming 384 strips
        maxPhiAdditional = cms.double(0.001096605744), #Assuming 384 strips
        maxETASeeds = cms.double(0.1), #Assuming 8 eta partitions
        maxTOFDiff = cms.double(25),
        requireCentralBX = cms.bool(True), #require that a majority of hits come from central BX
        minNumberOfHits = cms.uint32(4),
        maxNumberOfHits = cms.uint32(300),
        maxNumberOfHitsPerLayer = cms.uint32(100),
    ),
    algo_pset = cms.PSet(
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
