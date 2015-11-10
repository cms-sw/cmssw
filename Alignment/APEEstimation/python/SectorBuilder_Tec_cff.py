import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_cfi import *



##
## Whole Subdetector (means only one for both endcaps)
##

Tec = EmptySector.clone(
    name = 'Tec',
    subdetId = [6],
)

TEC = cms.VPSet(
    Tec,
)



##
## Separation of side(+,-)
##

TecMinus = Tec.clone(
    name = 'TecMinus',
    side = [1],
)
TecPlus = Tec.clone(
    name = 'TecPlus',
    side = [2],
)

TECSideSeparation = cms.VPSet(
    TecMinus,
    TecPlus,
)



##
## Separation of side + rings
##

TecMinusRing1 = TecMinus.clone(
    name = 'TecMinusRing1',
    ring = [1],
)
TecMinusRing2 = TecMinus.clone(
    name = 'TecMinusRing2',
    ring = [2],
)
TecMinusRing3 = TecMinus.clone(
    name = 'TecMinusRing3',
    ring = [3],
)
TecMinusRing4 = TecMinus.clone(
    name = 'TecMinusRing4',
    ring = [4],
)
TecMinusRing5 = TecMinus.clone(
    name = 'TecMinusRing5',
    ring = [5],
)
TecMinusRing6 = TecMinus.clone(
    name = 'TecMinusRing6',
    ring = [6],
)
TecMinusRing7 = TecMinus.clone(
    name = 'TecMinusRing7',
    ring = [7],
)
TecPlusRing1 = TecPlus.clone(
    name = 'TecPlusRing1',
    ring = [1],
)
TecPlusRing2 = TecPlus.clone(
    name = 'TecPlusRing2',
    ring = [2],
)
TecPlusRing3 = TecPlus.clone(
    name = 'TecPlusRing3',
    ring = [3],
)
TecPlusRing4 = TecPlus.clone(
    name = 'TecPlusRing4',
    ring = [4],
)
TecPlusRing5 = TecPlus.clone(
    name = 'TecPlusRing5',
    ring = [5],
)
TecPlusRing6 = TecPlus.clone(
    name = 'TecPlusRing6',
    ring = [6],
)
TecPlusRing7 = TecPlus.clone(
    name = 'TecPlusRing7',
    ring = [7],
)

TECSideAndPureRingSeparation = cms.VPSet(
    TecMinusRing1,
    TecMinusRing2,
    TecMinusRing3,
    TecMinusRing4,
    TecMinusRing5,
    TecMinusRing6,
    TecMinusRing7,
    TecPlusRing1,
    TecPlusRing2,
    TecPlusRing3,
    TecPlusRing4,
    TecPlusRing5,
    TecPlusRing6,
    TecPlusRing7,
)



##
## Separation of side + rings + rphi/stereo
##

TecMinusRing1Rphi = TecMinusRing1.clone(
    name = 'TecMinusRing1Rphi',
    isRPhi = [1],
)
TecMinusRing1Stereo = TecMinusRing1.clone(
    name = 'TecMinusRing1Stereo',
    isRPhi = [2],
)
TecMinusRing2Rphi = TecMinusRing2.clone(
    name = 'TecMinusRing2Rphi',
    isRPhi = [1],
)
TecMinusRing2Stereo = TecMinusRing2.clone(
    name = 'TecMinusRing2Stereo',
    isRPhi = [2],
)
TecMinusRing5Rphi = TecMinusRing5.clone(
    name = 'TecMinusRing5Rphi',
    isRPhi = [1],
)
TecMinusRing5Stereo = TecMinusRing5.clone(
    name = 'TecMinusRing5Stereo',
    isRPhi = [2],
)
TecPlusRing1Rphi = TecPlusRing1.clone(
    name = 'TecPlusRing1Rphi',
    isRPhi = [1],
)
TecPlusRing1Stereo = TecPlusRing1.clone(
    name = 'TecPlusRing1Stereo',
    isRPhi = [2],
)
TecPlusRing2Rphi = TecPlusRing2.clone(
    name = 'TecPlusRing2Rphi',
    isRPhi = [1],
)
TecPlusRing2Stereo = TecPlusRing2.clone(
    name = 'TecPlusRing2Stereo',
    isRPhi = [2],
)
TecPlusRing5Rphi = TecPlusRing5.clone(
    name = 'TecPlusRing5Rphi',
    isRPhi = [1],
)
TecPlusRing5Stereo = TecPlusRing5.clone(
    name = 'TecPlusRing5Stereo',
    isRPhi = [2],
)

TECSideAndRingSeparation = cms.VPSet(
    TecMinusRing1Rphi,
    TecMinusRing1Stereo,
    TecMinusRing2Rphi,
    TecMinusRing2Stereo,
    TecMinusRing3,
    TecMinusRing4,
    TecMinusRing5Rphi,
    TecMinusRing5Stereo,
    TecMinusRing6,
    TecMinusRing7,
    
    TecPlusRing1Rphi,
    TecPlusRing1Stereo,
    TecPlusRing2Rphi,
    TecPlusRing2Stereo,
    TecPlusRing3,
    TecPlusRing4,
    TecPlusRing5Rphi,
    TecPlusRing5Stereo,
    TecPlusRing6,
    TecPlusRing7,
)



##
## Separation of side + rings + rphi/stereo + orientations
##

TecMinusRing1RphiOut = TecMinusRing1Rphi.clone(
    name = 'TecMinusRing1RphiOut',
    wDirection = [-1],
)
TecMinusRing1StereoOut = TecMinusRing1Stereo.clone(
    name = 'TecMinusRing1StereoOut',
    wDirection = [-1],
)
TecMinusRing1RphiIn = TecMinusRing1Rphi.clone(
    name = 'TecMinusRing1RphiIn',
    wDirection = [1],
)
TecMinusRing1StereoIn = TecMinusRing1Stereo.clone(
    name = 'TecMinusRing1StereoIn',
    wDirection = [1],
)
TecMinusRing2RphiOut = TecMinusRing2Rphi.clone(
    name = 'TecMinusRing2RphiOut',
    wDirection = [-1],
)
TecMinusRing2StereoOut = TecMinusRing2Stereo.clone(
    name = 'TecMinusRing2StereoOut',
    wDirection = [-1],
)
TecMinusRing2RphiIn = TecMinusRing2Rphi.clone(
    name = 'TecMinusRing2RphiIn',
    wDirection = [1],
)
TecMinusRing2StereoIn = TecMinusRing2Stereo.clone(
    name = 'TecMinusRing2StereoIn',
    wDirection = [1],
)
TecMinusRing3Out = TecMinusRing3.clone(
    name = 'TecMinusRing3Out',
    wDirection = [-1],
)
TecMinusRing3In = TecMinusRing3.clone(
    name = 'TecMinusRing3In',
    wDirection = [1],
)
TecMinusRing4Out = TecMinusRing4.clone(
    name = 'TecMinusRing4Out',
    wDirection = [-1],
)
TecMinusRing4In = TecMinusRing4.clone(
    name = 'TecMinusRing4In',
    wDirection = [1],
)
TecMinusRing5RphiOut = TecMinusRing5Rphi.clone(
    name = 'TecMinusRing5RphiOut',
    wDirection = [-1],
)
TecMinusRing5StereoOut = TecMinusRing5Stereo.clone(
    name = 'TecMinusRing5StereoOut',
    wDirection = [-1],
)
TecMinusRing5RphiIn = TecMinusRing5Rphi.clone(
    name = 'TecMinusRing5RphiIn',
    wDirection = [1],
)
TecMinusRing5StereoIn = TecMinusRing5Stereo.clone(
    name = 'TecMinusRing5StereoIn',
    wDirection = [1],
)
TecMinusRing6Out = TecMinusRing6.clone(
    name = 'TecMinusRing6Out',
    wDirection = [-1],
)
TecMinusRing6In = TecMinusRing6.clone(
    name = 'TecMinusRing6In',
    wDirection = [1],
)
TecMinusRing7Out = TecMinusRing7.clone(
    name = 'TecMinusRing7Out',
    wDirection = [-1],
)
TecMinusRing7In = TecMinusRing7.clone(
    name = 'TecMinusRing7In',
    wDirection = [1],
)

TecPlusRing1RphiOut = TecPlusRing1Rphi.clone(
    name = 'TecPlusRing1RphiOut',
    wDirection = [1],
)
TecPlusRing1StereoOut = TecPlusRing1Stereo.clone(
    name = 'TecPlusRing1StereoOut',
    wDirection = [1],
)
TecPlusRing1RphiIn = TecPlusRing1Rphi.clone(
    name = 'TecPlusRing1RphiIn',
    wDirection = [-1],
)
TecPlusRing1StereoIn = TecPlusRing1Stereo.clone(
    name = 'TecPlusRing1StereoIn',
    wDirection = [-1],
)
TecPlusRing2RphiOut = TecPlusRing2Rphi.clone(
    name = 'TecPlusRing2RphiOut',
    wDirection = [1],
)
TecPlusRing2StereoOut = TecPlusRing2Stereo.clone(
    name = 'TecPlusRing2StereoOut',
    wDirection = [1],
)
TecPlusRing2RphiIn = TecPlusRing2Rphi.clone(
    name = 'TecPlusRing2RphiIn',
    wDirection = [-1],
)
TecPlusRing2StereoIn = TecPlusRing2Stereo.clone(
    name = 'TecPlusRing2StereoIn',
    wDirection = [-1],
)
TecPlusRing3Out = TecPlusRing3.clone(
    name = 'TecPlusRing3Out',
    wDirection = [1],
)
TecPlusRing3In = TecPlusRing3.clone(
    name = 'TecPlusRing3In',
    wDirection = [-1],
)
TecPlusRing4Out = TecPlusRing4.clone(
    name = 'TecPlusRing4Out',
    wDirection = [1],
)
TecPlusRing4In = TecPlusRing4.clone(
    name = 'TecPlusRing4In',
    wDirection = [-1],
)
TecPlusRing5RphiOut = TecPlusRing5Rphi.clone(
    name = 'TecPlusRing5RphiOut',
    wDirection = [1],
)
TecPlusRing5StereoOut = TecPlusRing5Stereo.clone(
    name = 'TecPlusRing5StereoOut',
    wDirection = [1],
)
TecPlusRing5RphiIn = TecPlusRing5Rphi.clone(
    name = 'TecPlusRing5RphiIn',
    wDirection = [-1],
)
TecPlusRing5StereoIn = TecPlusRing5Stereo.clone(
    name = 'TecPlusRing5StereoIn',
    wDirection = [-1],
)
TecPlusRing6Out = TecPlusRing6.clone(
    name = 'TecPlusRing6Out',
    wDirection = [1],
)
TecPlusRing6In = TecPlusRing6.clone(
    name = 'TecPlusRing6In',
    wDirection = [-1],
)
TecPlusRing7Out = TecPlusRing7.clone(
    name = 'TecPlusRing7Out',
    wDirection = [1],
)
TecPlusRing7In = TecPlusRing7.clone(
    name = 'TecPlusRing7In',
    wDirection = [-1],
)

# All RPhi modules within a ring point in same w direction. Same is valid for Stereo modules, but with opposite sign

TECSideAndRingAndOrientationSeparation = cms.VPSet(
    TecMinusRing1RphiOut,
    #TecMinusRing1StereoOut,
    #TecMinusRing1RphiIn,
    TecMinusRing1StereoIn,
    #TecMinusRing2RphiOut,
    TecMinusRing2StereoOut,
    TecMinusRing2RphiIn,
    #TecMinusRing2StereoIn,
    #TecMinusRing3Out,
    TecMinusRing3In,
    TecMinusRing4Out,
    #TecMinusRing4In,
    TecMinusRing5RphiOut,
    #TecMinusRing5StereoOut,
    #TecMinusRing5RphiIn,
    TecMinusRing5StereoIn,
    TecMinusRing6Out,
    #TecMinusRing6In,
    #TecMinusRing7Out,
    TecMinusRing7In,
    
    TecPlusRing1RphiOut,
    #TecPlusRing1StereoOut,
    #TecPlusRing1RphiIn,
    TecPlusRing1StereoIn,
    #TecPlusRing2RphiOut,
    TecPlusRing2StereoOut,
    TecPlusRing2RphiIn,
    #TecPlusRing2StereoIn,
    #TecPlusRing3Out,
    TecPlusRing3In,
    TecPlusRing4Out,
    #TecPlusRing4In,
    TecPlusRing5RphiOut,
    #TecPlusRing5StereoOut,
    #TecPlusRing5RphiIn,
    TecPlusRing5StereoIn,
    TecPlusRing6Out,
    #TecPlusRing6In,
    #TecPlusRing7Out,
    TecPlusRing7In,
)












