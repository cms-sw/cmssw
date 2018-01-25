# The purpose of this customize function is to switch the automatic
# pixel pair mitigation back to the "static" one

def customisePixelPairStaticMitigation(process):
    # Recovery for L2L3
    process.pixelPairStepSeedLayersB = process.pixelPairStepSeedLayers.clone(
        layerList = [
            'BPix1+BPix4',
        ]
    )
    from RecoTracker.TkTrackingRegions.pointSeededTrackingRegion_cfi import pointSeededTrackingRegion as _pointSeededTrackingRegion
    process.pixelPairStepTrackingRegionsB = _pointSeededTrackingRegion.clone(
        RegionPSet = dict(
            ptMin = 0.6,
            originRadius = 0.015,
            mode = "VerticesFixed",
            zErrorVetex = 0.03,
            vertexCollection = "firstStepPrimaryVertices",
            beamSpot = "offlineBeamSpot",
            maxNVertices = 5,
            maxNRegions = 5,
            whereToUseMeasurementTracker = "Never",
            deltaEta = 1.2,
            deltaPhi = 0.5,
            points = dict(
                eta = [0.0],
                phi = [3.0],
            )
        )
    )
    process.pixelPairStepHitDoubletsB.seedingLayers = "pixelPairStepSeedLayersB"
    process.pixelPairStepHitDoubletsB.trackingRegions = "pixelPairStepTrackingRegionsB"
    process.pixelPairStepHitDoubletsB.trackingRegionsSeedingLayers = ""

    process.PixelPairStepTask.remove(process.pixelPairStepTrackingRegionsSeedLayersB)
    process.PixelPairStepTask.add(process.pixelPairStepSeedLayersB,
                                  process.pixelPairStepTrackingRegionsB)


    # Adjust DQM as well if exists
    if hasattr(process, "TrackSeedMonpixelPairStep"):
        process.TrackSeedMonpixelPairStep.doRegionPlots = False
        process.TrackSeedMonpixelPairStep.RegionSeedingLayersProducer = ""

    return process
