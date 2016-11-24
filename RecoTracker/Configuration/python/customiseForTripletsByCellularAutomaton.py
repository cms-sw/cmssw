import FWCore.ParameterSet.Config as cms

_CAParameters = dict(
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 20, value2 = 10,
        enabled = True,
    ),
    useBendingCorrection = True,
    CAThetaCut = 0.0015,
    CAPhiCut = 0.01,
    CAHardPtCut = 0,
)

def customiseLegacySeeding(module):
    pset = getattr(module, "OrderedHitsFactoryPSet")
    if not hasattr(pset, "ComponentName"):
        return
    if pset.ComponentName != "StandardHitTripletGenerator":
        return
    # Adjust seeding layers
    seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()


    # Configure seed generator / pixel track producer
    Triplets = module.OrderedHitsFactoryPSet.clone()
    from RecoPixelVertexing.PixelTriplets.CAHitTripletGenerator_cfi import CAHitTripletGenerator as _CAHitTripletGenerator

    module.OrderedHitsFactoryPSet  = _CAHitTripletGenerator.clone(
        ComponentName = "CAHitTripletGenerator",
        extraHitRPhitolerance = Triplets.GeneratorPSet.extraHitRPhitolerance,
        SeedingLayers = seedingLayersName,
        **_CAParameters
    )

    if hasattr(Triplets.GeneratorPSet, "SeedComparitorPSet"):
        module.OrderedHitsFactoryPSet.SeedComparitorPSet = Triplets.GeneratorPSet.SeedComparitorPSet

def customiseNewSeeding(process, module):
    doubletModuleName = module.doublets.getModuleLabel()
    doubletModule = getattr(process, doubletModuleName)

    # Generate doublets for all adjacent layer pairs
    doubletModule.layerPairs = [
        0, # layer pair (0,1)
        1, # layer pair (1,2)
    ]

    # Bit of a hack to replace a module with another, but works
    #
    # In principle setattr(process) could work too, but it expands the
    # sequences and I don't want that
    modifier = cms.Modifier()
    modifier._setChosen()

    comparitor = None
    if hasattr(module, "SeedComparitorPSet"):
        comparitor = module.SeedComparitorPSet.clone()

    # Replace triplet generator with the CA version
    from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer
    modifier.toReplaceWith(module, _caHitTripletEDProducer.clone(
        doublets = doubletModuleName,
        extraHitRPhitolerance = module.extraHitRPhitolerance,
        **_CAParameters
    ))

    if comparitor:
        module.SeedComparitorPSet = comparitor


def customiseForTripletsByCellularAutomaton(process):
    for module in process._Process__producers.values():
        if hasattr(module, "OrderedHitsFactoryPSet"):
            customiseLegacySeeding(module)
        elif module._TypedParameterizable__type in ["PixelTripletHLTEDProducer", "PixelTripletLargeTipEDProducer"]:
            customiseNewSeeding(process, module)
    return process
