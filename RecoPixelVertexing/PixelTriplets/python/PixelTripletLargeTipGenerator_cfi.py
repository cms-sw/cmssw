import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
eras.trackingPhase1PU70.toModify(PixelTripletLargeTipGenerator, maxElement = 0)

