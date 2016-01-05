import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_cfi import *



##
## Whole Subdetector
##

Tob = EmptySector.clone(
    name = 'Tob',
    subdetId = [5],
)

TOB = cms.VPSet(
    Tob,
)



##
##  cosmic-like quartering (upper, lower, left, right part)
##

TobUp = Tob.clone(
    name = 'TobUp',
    posPhi = [0.7854,2.3562],   # [45,135] degree
)
TobDown = Tob.clone(
    name = 'TobDown',
    posPhi = [-2.3562,-0.7854],   # [-135,-45] degree
)
TobLeft = Tob.clone(
    name = 'TobLeft',
    posPhi = [-0.7854,0.7854],   # [-45,45] degree
)
TobRight = Tob.clone(
    name = 'TobRight',
    posPhi = [-3.5,-2.3562,2.3562,3.5],   # [135,-135] degree
)

TOBQuarters = cms.VPSet(
    TobUp,
    TobDown,
    TobLeft,
    TobRight,
)



##
## Separation of pitches + Separation of 1D and 2D layers
##

TobLayer12 = Tob.clone(
    name = 'TobLayer12',
    layer = [1,2],
)
TobLayer34 = Tob.clone(
    name = 'TobLayer34',
    layer = [3,4],
)
TobLayer56 = Tob.clone(
    name = 'TobLayer56',
    layer = [5,6],
)

TOBPitchAnd2DSeparation = cms.VPSet(
    TobLayer12,
    TobLayer34,
    TobLayer56,
)



##
## Separation of layers
##

TobLayer1 = Tob.clone(
    name = 'TobLayer1',
    layer = [1],
)
TobLayer2 = Tob.clone(
    name = 'TobLayer2',
    layer = [2],
)
TobLayer3 = Tob.clone(
    name = 'TobLayer3',
    layer = [3],
)
TobLayer4 = Tob.clone(
    name = 'TobLayer4',
    layer = [4],
)
TobLayer5 = Tob.clone(
    name = 'TobLayer5',
    layer = [5],
)
TobLayer6 = Tob.clone(
    name = 'TobLayer6',
    layer = [6],
)

TOBPureLayerSeparation = cms.VPSet(
    TobLayer1,
    TobLayer2,
    TobLayer3,
    TobLayer4,
    TobLayer5,
    TobLayer6,
)



##
## Separation of layers + rphi/stereo
##

TobLayer1Rphi = TobLayer1.clone(
    name = 'TobLayer1Rphi',
    isRPhi = [1],
)
TobLayer1Stereo = TobLayer1.clone(
    name = 'TobLayer1Stereo',
    isRPhi = [2],
)
TobLayer2Rphi = TobLayer2.clone(
    name = 'TobLayer2Rphi',
    isRPhi = [1],
)
TobLayer2Stereo = TobLayer2.clone(
    name = 'TobLayer2Stereo',
    isRPhi = [2],
)

TOBLayerSeparation = cms.VPSet(
    TobLayer1Rphi,
    TobLayer1Stereo,
    TobLayer2Rphi,
    TobLayer2Stereo,
    TobLayer3,
    TobLayer4,
    TobLayer5,
    TobLayer6,
)



##
## Separation of layers + rphi/stereo + orientations
##

TobLayer1RphiOut = TobLayer1Rphi.clone(
    name = 'TobLayer1RphiOut',
    wDirection = [1],
)
TobLayer1StereoOut = TobLayer1Stereo.clone(
    name = 'TobLayer1StereoOut',
    wDirection = [1],
)
TobLayer1RphiIn = TobLayer1Rphi.clone(
    name = 'TobLayer1RphiIn',
    wDirection = [-1],
)
TobLayer1StereoIn = TobLayer1Stereo.clone(
    name = 'TobLayer1StereoIn',
    wDirection = [-1],
)
TobLayer2RphiOut = TobLayer2Rphi.clone(
    name = 'TobLayer2RphiOut',
    wDirection = [1],
)
TobLayer2StereoOut = TobLayer2Stereo.clone(
    name = 'TobLayer2StereoOut',
    wDirection = [1],
)
TobLayer2RphiIn = TobLayer2Rphi.clone(
    name = 'TobLayer2RphiIn',
    wDirection = [-1],
)
TobLayer2StereoIn = TobLayer2Stereo.clone(
    name = 'TobLayer2StereoIn',
    wDirection = [-1],
)
TobLayer3Out = TobLayer3.clone(
    name = 'TobLayer3Out',
    wDirection = [1],
)
TobLayer3In = TobLayer3.clone(
    name = 'TobLayer3In',
    wDirection = [-1],
)
TobLayer4Out = TobLayer4.clone(
    name = 'TobLayer4Out',
    wDirection = [1],
)
TobLayer4In = TobLayer4.clone(
    name = 'TobLayer4In',
    wDirection = [-1],
)
TobLayer5Out = TobLayer5.clone(
    name = 'TobLayer5Out',
    wDirection = [1],
)
TobLayer5In = TobLayer5.clone(
    name = 'TobLayer5In',
    wDirection = [-1],
)
TobLayer6Out = TobLayer6.clone(
    name = 'TobLayer6Out',
    wDirection = [1],
)
TobLayer6In = TobLayer6.clone(
    name = 'TobLayer6In',
    wDirection = [-1],
)

# All RPhi modules within a layer point in same w direction. Same is valid for Stereo modules, but with opposite sign

TOBLayerAndOrientationSeparation = cms.VPSet(
    #TobLayer1RphiOut,      # no modules contained
    TobLayer1StereoOut,
    TobLayer1RphiIn,
    #TobLayer1StereoIn,     # no modules contained
    TobLayer2RphiOut,
    #TobLayer2StereoOut,    # no modules contained
    #TobLayer2RphiIn,       # no modules contained
    TobLayer2StereoIn,
    TobLayer3Out,
    TobLayer3In,
    TobLayer4Out,
    TobLayer4In,
    TobLayer5Out,
    TobLayer5In,
    TobLayer6Out,
    TobLayer6In,
)






  









