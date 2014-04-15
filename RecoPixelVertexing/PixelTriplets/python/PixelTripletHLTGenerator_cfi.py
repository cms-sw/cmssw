import FWCore.ParameterSet.Config as cms

# moving to the block.  Will delete the PSet once transition is done
PixelTripletHLTGenerator = cms.PSet(
   maxElement = cms.uint32(100000),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    extraHitRPhitolerance = cms.double(0.032),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3), ## can be removed if !useFixedPreFiltering
    
    #double extraHitRZtolerance   = 0.207     # ok for strips
    #double extraHitRPhitolerance = 0.102     # ok for strips
    extraHitRZtolerance = cms.double(0.037),
    SeedComparitorPSet = cms.PSet(
     ComponentName = cms.string('none')
     )
)

import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
PixelTripletHLTGeneratorWithFilter = cms.PSet(
   maxElement = cms.uint32(100000),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    extraHitRPhitolerance = cms.double(0.032),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3), ## can be removed if !useFixedPreFiltering
    
    #double extraHitRZtolerance   = 0.207     # ok for strips
    #double extraHitRPhitolerance = 0.102     # ok for strips
    extraHitRZtolerance = cms.double(0.037),
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
)

