def customiseClusterCheckForHighPileup(process):
    for module in process._Process__producers.values():
        if hasattr(module, "ClusterCheckPSet"):
            module.ClusterCheckPSet.MaxNumberOfPixelClusters = 80000
            # PhotonConversionTrajectorySeedProducerFromQuadruplets does not have "cut"...
            if hasattr(module.ClusterCheckPSet, "cut"):
                module.ClusterCheckPSet.cut = "strip < 800000 && pixel < 80000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/7.)"
        if hasattr(module, "OrderedHitsFactoryPSet") and hasattr(module.OrderedHitsFactoryPSet, "GeneratorPSet"):
            if module.OrderedHitsFactoryPSet.GeneratorPSet.ComponentName.value() in ["PixelTripletLargeTipGenerator", "MultiHitGeneratorFromChi2"]:
                module.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 1000000

    return process
