def customiseClusterCheckForHighPileup(process):
    _maxPixel = 80000
    _cut = "strip < 800000 && pixel < 80000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/7.)"

    _maxElement = 1000000

    for module in process._Process__producers.values():
        cppType = module._TypedParameterizable__type

        # cluster multiplicity check
        if cppType == "ClusterCheckerEDProducer":
            module.MaxNumberOfPixelClusters = _maxPixel
            module.cut = _cut
        if hasattr(module, "ClusterCheckPSet"):
            module.ClusterCheckPSet.MaxNumberOfPixelClusters = _maxPixel
            # PhotonConversionTrajectorySeedProducerFromQuadruplets does not have "cut"...
            if hasattr(module.ClusterCheckPSet, "cut"):
                module.ClusterCheckPSet.cut = _cut

        # maxElement
        if cppType in ["PixelTripletLargeTipEDProducer", "MultiHitFromChi2EDProducer"]:
            module.maxElement = _maxElement
        if hasattr(module, "OrderedHitsFactoryPSet") and hasattr(module.OrderedHitsFactoryPSet, "GeneratorPSet"):
            if module.OrderedHitsFactoryPSet.GeneratorPSet.ComponentName.value() in ["PixelTripletLargeTipGenerator", "MultiHitGeneratorFromChi2"]:
                module.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = _maxElement

    return process
