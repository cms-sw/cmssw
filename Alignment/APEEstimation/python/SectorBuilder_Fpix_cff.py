import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_cfi import *



##
## Whole Subdetector (means only one for both endcaps)
##

Fpix = EmptySector.clone(
    name = 'Fpix',
    subdetId = [2],
)

FPIX = cms.VPSet(
    Fpix,
)



##
## Separation of side(+,-)
##

FpixMinus = Fpix.clone(
    name = 'FpixMinus',
    side = [1],
)
FpixPlus = Fpix.clone(
    name = 'FpixPlus',
    side = [2],
)

FPIXSideSeparation = cms.VPSet(
    FpixMinus,
    FpixPlus,
)



##
## Separation of side + layers
##

FpixMinusLayer1 = FpixMinus.clone(
    name = 'FpixMinusLayer1',
    layer = [1],
)
FpixMinusLayer2 = FpixMinus.clone(
    name = 'FpixMinusLayer2',
    layer = [2],
)
FpixMinusLayer3 = FpixMinus.clone(
    name = 'FpixMinusLayer3',
    layer = [3],
)
FpixPlusLayer1 = FpixPlus.clone(
    name = 'FpixPlusLayer1',
    layer = [1],
)
FpixPlusLayer2 = FpixPlus.clone(
    name = 'FpixPlusLayer2',
    layer = [2],
)
FpixPlusLayer3 = FpixPlus.clone(
    name = 'FpixPlusLayer3',
    layer = [3],
)

FPIXSideAndLayerSeparation = cms.VPSet(
    FpixMinusLayer1,
    FpixMinusLayer2,
    FpixMinusLayer3,
    FpixPlusLayer1,
    FpixPlusLayer2,
    FpixPlusLayer3,
)



##
## Separation of side + layers + orientations
##

FpixMinusLayer1Out = FpixMinusLayer1.clone(
    name = 'FpixMinusLayer1Out',
    wDirection = [-1],
)
FpixMinusLayer1In = FpixMinusLayer1.clone(
    name = 'FpixMinusLayer1In',
    wDirection = [1],
)
FpixMinusLayer2Out = FpixMinusLayer2.clone(
    name = 'FpixMinusLayer2Out',
    wDirection = [-1],
)
FpixMinusLayer2In = FpixMinusLayer2.clone(
    name = 'FpixMinusLayer2In',
    wDirection = [1],
)
FpixMinusLayer3Out = FpixMinusLayer3.clone(
    name = 'FpixMinusLayer3Out',
    wDirection = [-1],
)
FpixMinusLayer3In = FpixMinusLayer3.clone(
    name = 'FpixMinusLayer3In',
    wDirection = [1],
)
FpixPlusLayer1Out = FpixPlusLayer1.clone(
    name = 'FpixPlusLayer1Out',
    wDirection = [1],
)
FpixPlusLayer1In = FpixPlusLayer1.clone(
    name = 'FpixPlusLayer1In',
    wDirection = [-1],
)
FpixPlusLayer2Out = FpixPlusLayer2.clone(
    name = 'FpixPlusLayer2Out',
    wDirection = [1],
)
FpixPlusLayer2In = FpixPlusLayer2.clone(
    name = 'FpixPlusLayer2In',
    wDirection = [-1],
)
FpixPlusLayer3Out = FpixPlusLayer3.clone(
    name = 'FpixPlusLayer3Out',
    wDirection = [1],
)
FpixPlusLayer3In = FpixPlusLayer3.clone(
    name = 'FpixPlusLayer3In',
    wDirection = [-1],
)

FPIXSideAndLayerAndOrientationSeparation = cms.VPSet(
    FpixMinusLayer1Out,
    FpixMinusLayer1In,
    FpixMinusLayer2Out,
    FpixMinusLayer2In,
    FpixMinusLayer3Out,
    FpixMinusLayer3In,
    FpixPlusLayer1Out,
    FpixPlusLayer1In,
    FpixPlusLayer2Out,
    FpixPlusLayer2In,
    FpixPlusLayer3Out,
    FpixPlusLayer3In,
)








