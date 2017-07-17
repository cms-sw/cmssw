import FWCore.ParameterSet.Config as cms

_CAParameters = dict(
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 200, value2 = 100,
        enabled = True,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.00125,
    CAPhiCut = 0.1,
    CAHardPtCut = 0,
)

def customiseLegacySeeding(module):
    pset = getattr(module, "OrderedHitsFactoryPSet")
    if not hasattr(pset, "ComponentName"):
        return
    if pset.ComponentName != "CombinedHitQuadrupletGenerator":
        return
    # Adjust seeding layers
    seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
   
    # Configure seed generator / pixel track producer
    quadruplets = module.OrderedHitsFactoryPSet.clone()
    from RecoPixelVertexing.PixelTriplets.CAHitQuadrupletGenerator_cfi import CAHitQuadrupletGenerator as _CAHitQuadrupletGenerator

    module.OrderedHitsFactoryPSet = _CAHitQuadrupletGenerator.clone(
        ComponentName = "CAHitQuadrupletGenerator",
        extraHitRPhitolerance = quadruplets.GeneratorPSet.extraHitRPhitolerance,
        SeedingLayers = seedingLayersName,
        **_CAParameters
    )

    if hasattr(quadruplets.GeneratorPSet, "SeedComparitorPSet"):
        module.OrderedHitsFactoryPSet.SeedComparitorPSet = quadruplets.GeneratorPSet.SeedComparitorPSet

def customiseNewSeeding(process, module):
    tripletModuleName = module.triplets.getModuleLabel()
    tripletModule = getattr(process, tripletModuleName)
    doubletModuleName = tripletModule.doublets.getModuleLabel()
    doubletModule = getattr(process, doubletModuleName)

    # Generate doublets for all adjacent layer pairs
    doubletModule.layerPairs = [
        0, # layer pair (0,1)
        1, # layer pair (1,2)
        2  # layer pair (2,3)
    ]

    # Bit of a hack to replace a module with another, but works
    #
    # In principle setattr(process) could work too, but it expands the
    # sequences and I don't want that
    modifier = cms.Modifier()
    modifier._setChosen()

    # Replace quadruplet generator with the CA version
    from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
    modifier.toReplaceWith(module, _caHitQuadrupletEDProducer.clone(
        doublets = doubletModuleName,
        SeedComparitorPSet = module.SeedComparitorPSet.clone(),
        extraHitRPhitolerance = module.extraHitRPhitolerance,
        **_CAParameters
    ))

    # Remove triplet generator from all sequence and paths
    for seqs in [process.sequences_(), process.paths_()]:
        for seqName, seq in seqs.iteritems():
            # cms.Sequence.remove() would look simpler, but it expands
            # the contained sequences if a replacement occurs there.
            try:
                index = seq.index(tripletModule)
            except:
                continue

            seq.remove(tripletModule)

    delattr(process, tripletModuleName)

def customiseForQuadrupletsByCellularAutomaton(process):
    for module in process._Process__producers.values():
        if hasattr(module, "OrderedHitsFactoryPSet"):
            customiseLegacySeeding(module)
        elif module._TypedParameterizable__type == "PixelQuadrupletEDProducer":
            customiseNewSeeding(process, module)

    return process
