def customiseClusterCheckForHighPileup(process):
    for module in process._Process__producers.values():
        if hasattr(module, "ClusterCheckPSet"):
            module.ClusterCheckPSet.MaxNumberOfPixelClusters = 400000
            # PhotonConversionTrajectorySeedProducerFromQuadruplets does not have "cut"...
            if hasattr(module.ClusterCheckPSet, "cut"):
                module.ClusterCheckPSet.cut = "strip < 4000000 && pixel < 400000 && (strip < 500000 + 10*pixel) && (pixel < 50000 + 0.1*strip)"

    return process
