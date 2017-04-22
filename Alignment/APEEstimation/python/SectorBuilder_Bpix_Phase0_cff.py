import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_cfi import *



##
## Whole Subdetector
##

Bpix = EmptySector.clone(
    name = 'Bpix',
    subdetId = [1],
)

BPIX = cms.VPSet(
    Bpix,
)



##
## Separation of layers
##

BpixLayer1 = Bpix.clone(
    name = 'BpixLayer1',
    layer = [1],
)
BpixLayer2 = Bpix.clone(
    name = 'BpixLayer2',
    layer = [2],
)
BpixLayer3 = Bpix.clone(
    name = 'BpixLayer3',
    layer = [3],
)

BPIXLayerSeparation = cms.VPSet(
    BpixLayer1,
    BpixLayer2,
    BpixLayer3,
)



##
## Separation of layers + orientations
##

BpixLayer1Out = BpixLayer1.clone(
    name = 'BpixLayer1Out',
    wDirection = [1],
)
BpixLayer1In = BpixLayer1.clone(
    name = 'BpixLayer1In',
    wDirection = [-1],
)
BpixLayer2Out = BpixLayer2.clone(
    name = 'BpixLayer2Out',
    wDirection = [1],
)
BpixLayer2In = BpixLayer2.clone(
    name = 'BpixLayer2In',
    wDirection = [-1],
)
BpixLayer3Out = BpixLayer3.clone(
    name = 'BpixLayer3Out',
    wDirection = [1],
)
BpixLayer3In = BpixLayer3.clone(
    name = 'BpixLayer3In',
    wDirection = [-1],
)

BPIXLayerAndOrientationSeparation = cms.VPSet(
    BpixLayer1Out,
    BpixLayer1In,
    BpixLayer2Out,
    BpixLayer2In,
    BpixLayer3Out,
    BpixLayer3In,
)














