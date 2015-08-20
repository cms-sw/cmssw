import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_cfi import *



##
## Whole Subdetector
##

Tib = EmptySector.clone(
    name = 'Tib',
    subdetId = [3],
)

TIB = cms.VPSet(
    Tib,
)



##
##  cosmic-like quartering (upper, lower, left, right part)
##

TibUp = Tib.clone(
    name = 'TibUp',
    posPhi = [0.7854,2.3562],   # [45,135] degree
)
TibDown = Tib.clone(
    name = 'TibDown',
    posPhi = [-2.3562,-0.7854],   # [-135,-45] degree
)
TibLeft = Tib.clone(
    name = 'TibLeft',
    posPhi = [-0.7854,0.7854],   # [-45,45] degree
)
TibRight = Tib.clone(
    name = 'TibRight',
    posPhi = [-3.5,-2.3562,2.3562,3.5],   # [135,-135] degree
)

TIBQuarters = cms.VPSet(
    TibUp,
    TibDown,
    TibLeft,
    TibRight,
)



##
## Separation of pitches + Separation of 1D and 2D layers
##

TibLayer12 = Tib.clone(
    name = 'TibLayer12',
    layer = [1,2],
)
TibLayer34 = Tib.clone(
    name = 'TibLayer34',
    layer = [3,4],
)

TIBPitchAnd2DSeparation = cms.VPSet(
    TibLayer12,
    TibLayer34,
)



##
## Separation of layers
##

TibLayer1 = Tib.clone(
    name = 'TibLayer1',
    layer = [1],
)
TibLayer2 = Tib.clone(
    name = 'TibLayer2',
    layer = [2],
)
TibLayer3 = Tib.clone(
    name = 'TibLayer3',
    layer = [3],
)
TibLayer4 = Tib.clone(
    name = 'TibLayer4',
    layer = [4],
)

TIBPureLayerSeparation = cms.VPSet(
    TibLayer1,
    TibLayer2,
    TibLayer3,
    TibLayer4,
)



##
## Separation of layers + rphi/stereo
##

TibLayer1Rphi = TibLayer1.clone(
    name = 'TibLayer1Rphi',
    isRPhi = [1],
)
TibLayer1Stereo = TibLayer1.clone(
    name = 'TibLayer1Stereo',
    isRPhi = [2],
)
TibLayer2Rphi = TibLayer2.clone(
    name = 'TibLayer2Rphi',
    isRPhi = [1],
)
TibLayer2Stereo = TibLayer2.clone(
    name = 'TibLayer2Stereo',
    isRPhi = [2],
)

TIBLayerSeparation = cms.VPSet(
    TibLayer1Rphi,
    TibLayer1Stereo,
    TibLayer2Rphi,
    TibLayer2Stereo,
    TibLayer3,
    TibLayer4,
)



##
## Separation of layers + rphi/stereo + orientations
##

TibLayer1RphiOut = TibLayer1Rphi.clone(
    name = 'TibLayer1RphiOut',
    wDirection = [1],
)
TibLayer1StereoOut = TibLayer1Stereo.clone(
    name = 'TibLayer1StereoOut',
    wDirection = [1],
)
TibLayer1RphiIn = TibLayer1Rphi.clone(
    name = 'TibLayer1RphiIn',
    wDirection = [-1],
)
TibLayer1StereoIn = TibLayer1Stereo.clone(
    name = 'TibLayer1StereoIn',
    wDirection = [-1],
)
TibLayer2RphiOut = TibLayer2Rphi.clone(
    name = 'TibLayer2RphiOut',
    wDirection = [1],
)
TibLayer2StereoOut = TibLayer2Stereo.clone(
    name = 'TibLayer2StereoOut',
    wDirection = [1],
)
TibLayer2RphiIn = TibLayer2Rphi.clone(
    name = 'TibLayer2RphiIn',
    wDirection = [-1],
)
TibLayer2StereoIn = TibLayer2Stereo.clone(
    name = 'TibLayer2StereoIn',
    wDirection = [-1],
)
TibLayer3Out = TibLayer3.clone(
    name = 'TibLayer3Out',
    wDirection = [1],
)
TibLayer3In = TibLayer3.clone(
    name = 'TibLayer3In',
    wDirection = [-1],
)
TibLayer4Out = TibLayer4.clone(
    name = 'TibLayer4Out',
    wDirection = [1],
)
TibLayer4In = TibLayer4.clone(
    name = 'TibLayer4In',
    wDirection = [-1],
)

TIBLayerAndOrientationSeparation = cms.VPSet(
    TibLayer1RphiOut,
    TibLayer1StereoOut,
    TibLayer1RphiIn,
    TibLayer1StereoIn,
    TibLayer2RphiOut,
    TibLayer2StereoOut,
    TibLayer2RphiIn,
    TibLayer2StereoIn,
    TibLayer3Out,
    TibLayer3In,
    TibLayer4Out,
    TibLayer4In,
)














