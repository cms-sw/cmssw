import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_cfi import *



##
## Whole Subdetector (means only one for both endcaps)
##

Tid = EmptySector.clone(
    name = 'Tid',
    subdetId = [4],
)

TID = cms.VPSet(
    Tid,
)



##
## Separation of side(+,-)
##

TidMinus = Tid.clone(
    name = 'TidMinus',
    side = [1],
)
TidPlus = Tid.clone(
    name = 'TidPlus',
    side = [2],
)

TIDSideSeparation = cms.VPSet(
    TidMinus,
    TidPlus,
)



##
## Separation of side + rings
##

TidMinusRing1 = TidMinus.clone(
    name = 'TidMinusRing1',
    ring = [1],
)
TidMinusRing2 = TidMinus.clone(
    name = 'TidMinusRing2',
    ring = [2],
)
TidMinusRing3 = TidMinus.clone(
    name = 'TidMinusRing3',
    ring = [3],
)
TidPlusRing1 = TidPlus.clone(
    name = 'TidPlusRing1',
    ring = [1],
)
TidPlusRing2 = TidPlus.clone(
    name = 'TidPlusRing2',
    ring = [2],
)
TidPlusRing3 = TidPlus.clone(
    name = 'TidPlusRing3',
    ring = [3],
)

TIDSideAndPureRingSeparation = cms.VPSet(
    TidMinusRing1,
    TidMinusRing2,
    TidMinusRing3,
    TidPlusRing1,
    TidPlusRing2,
    TidPlusRing3,
)



##
## Separation of side + rings + rphi/stereo
##

TidMinusRing1Rphi = TidMinusRing1.clone(
    name = 'TidMinusRing1Rphi',
    isRPhi = [1],
)
TidMinusRing1Stereo = TidMinusRing1.clone(
    name = 'TidMinusRing1Stereo',
    isRPhi = [2],
)
TidMinusRing2Rphi = TidMinusRing2.clone(
    name = 'TidMinusRing2Rphi',
    isRPhi = [1],
)
TidMinusRing2Stereo = TidMinusRing2.clone(
    name = 'TidMinusRing2Stereo',
    isRPhi = [2],
)
TidPlusRing1Rphi = TidPlusRing1.clone(
    name = 'TidPlusRing1Rphi',
    isRPhi = [1],
)
TidPlusRing1Stereo = TidPlusRing1.clone(
    name = 'TidPlusRing1Stereo',
    isRPhi = [2],
)
TidPlusRing2Rphi = TidPlusRing2.clone(
    name = 'TidPlusRing2Rphi',
    isRPhi = [1],
)
TidPlusRing2Stereo = TidPlusRing2.clone(
    name = 'TidPlusRing2Stereo',
    isRPhi = [2],
)

TIDSideAndRingSeparation = cms.VPSet(
    TidMinusRing1Rphi,
    TidMinusRing1Stereo,
    TidMinusRing2Rphi,
    TidMinusRing2Stereo,
    TidMinusRing3,
    
    TidPlusRing1Rphi,
    TidPlusRing1Stereo,
    TidPlusRing2Rphi,
    TidPlusRing2Stereo,
    TidPlusRing3,
)



##
## Separation of side + rings + rphi/stereo + orientations
##

TidMinusRing1RphiOut = TidMinusRing1Rphi.clone(
    name = 'TidMinusRing1RphiOut',
    wDirection = [-1],
)
TidMinusRing1StereoOut = TidMinusRing1Stereo.clone(
    name = 'TidMinusRing1StereoOut',
    wDirection = [-1],
)
TidMinusRing1RphiIn = TidMinusRing1Rphi.clone(
    name = 'TidMinusRing1RphiIn',
    wDirection = [1],
)
TidMinusRing1StereoIn = TidMinusRing1Stereo.clone(
    name = 'TidMinusRing1StereoIn',
    wDirection = [1],
)
TidMinusRing2RphiOut = TidMinusRing2Rphi.clone(
    name = 'TidMinusRing2RphiOut',
    wDirection = [-1],
)
TidMinusRing2StereoOut = TidMinusRing2Stereo.clone(
    name = 'TidMinusRing2StereoOut',
    wDirection = [-1],
)
TidMinusRing2RphiIn = TidMinusRing2Rphi.clone(
    name = 'TidMinusRing2RphiIn',
    wDirection = [1],
)
TidMinusRing2StereoIn = TidMinusRing2Stereo.clone(
    name = 'TidMinusRing2StereoIn',
    wDirection = [1],
)
TidMinusRing3Out = TidMinusRing3.clone(
    name = 'TidMinusRing3Out',
    wDirection = [-1],
)
TidMinusRing3In = TidMinusRing3.clone(
    name = 'TidMinusRing3In',
    wDirection = [1],
)

TidPlusRing1RphiOut = TidPlusRing1Rphi.clone(
    name = 'TidPlusRing1RphiOut',
    wDirection = [1],
)
TidPlusRing1StereoOut = TidPlusRing1Stereo.clone(
    name = 'TidPlusRing1StereoOut',
    wDirection = [1],
)
TidPlusRing1RphiIn = TidPlusRing1Rphi.clone(
    name = 'TidPlusRing1RphiIn',
    wDirection = [-1],
)
TidPlusRing1StereoIn = TidPlusRing1Stereo.clone(
    name = 'TidPlusRing1StereoIn',
    wDirection = [-1],
)
TidPlusRing2RphiOut = TidPlusRing2Rphi.clone(
    name = 'TidPlusRing2RphiOut',
    wDirection = [1],
)
TidPlusRing2StereoOut = TidPlusRing2Stereo.clone(
    name = 'TidPlusRing2StereoOut',
    wDirection = [1],
)
TidPlusRing2RphiIn = TidPlusRing2Rphi.clone(
    name = 'TidPlusRing2RphiIn',
    wDirection = [-1],
)
TidPlusRing2StereoIn = TidPlusRing2Stereo.clone(
    name = 'TidPlusRing2StereoIn',
    wDirection = [-1],
)
TidPlusRing3Out = TidPlusRing3.clone(
    name = 'TidPlusRing3Out',
    wDirection = [1],
)
TidPlusRing3In = TidPlusRing3.clone(
    name = 'TidPlusRing3In',
    wDirection = [-1],
)

TIDSideAndRingAndOrientationSeparation = cms.VPSet(
    TidMinusRing1RphiOut,
    TidMinusRing1StereoOut,
    TidMinusRing1RphiIn,
    TidMinusRing1StereoIn,
    TidMinusRing2RphiOut,
    TidMinusRing2StereoOut,
    TidMinusRing2RphiIn,
    TidMinusRing2StereoIn,
    TidMinusRing3Out,
    TidMinusRing3In,
    
    TidPlusRing1RphiOut,
    TidPlusRing1StereoOut,
    TidPlusRing1RphiIn,
    TidPlusRing1StereoIn,
    TidPlusRing2RphiOut,
    TidPlusRing2StereoOut,
    TidPlusRing2RphiIn,
    TidPlusRing2StereoIn,
    TidPlusRing3Out,
    TidPlusRing3In,
)












