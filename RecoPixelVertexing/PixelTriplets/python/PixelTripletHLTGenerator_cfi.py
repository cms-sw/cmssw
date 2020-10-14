import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140

# moving to the block.  Will delete the PSet once transition is done
PixelTripletHLTGenerator = cms.PSet(
   maxElement = cms.uint32(100000),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    extraHitRPhitolerance = cms.double(0.032),
    extraHitRZtolerance = cms.double(0.037),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3),
    SeedComparitorPSet = cms.PSet(
     ComponentName = cms.string('none')
     )
)

# do thy make any difference anywhere?
trackingPhase2PU140.toModify(PixelTripletHLTGenerator,
    extraHitRPhitolerance = 0.016,
    extraHitRZtolerance   = 0.020
)

import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
PixelTripletHLTGeneratorWithFilter = PixelTripletHLTGenerator.clone(
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
)

