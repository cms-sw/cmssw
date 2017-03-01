import FWCore.ParameterSet.Config as cms

# moving to the block.  Will delete the PSet once transition is done
PixelTripletLargeTipGenerator = cms.PSet(
    maxElement = cms.uint32(100000),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    ComponentName = cms.string('PixelTripletLargeTipGenerator'),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3),
    extraHitRPhitolerance = cms.double(0.032),
    extraHitRZtolerance = cms.double(0.037)
)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(PixelTripletLargeTipGenerator, maxElement = 0)

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
peripheralPbPb.toModify(PixelTripletLargeTipGenerator, maxElement = 1000000)
