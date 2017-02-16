import FWCore.ParameterSet.Config as cms

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

import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
PixelTripletHLTGeneratorWithFilter = PixelTripletHLTGenerator.clone()
PixelTripletHLTGeneratorWithFilter.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()


