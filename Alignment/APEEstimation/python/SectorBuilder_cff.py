import FWCore.ParameterSet.Config as cms

from Alignment.APEEstimation.SectorBuilder_Bpix_cff import *
from Alignment.APEEstimation.SectorBuilder_Fpix_cff import *
from Alignment.APEEstimation.SectorBuilder_Tib_cff import *
from Alignment.APEEstimation.SectorBuilder_Tid_cff import *
from Alignment.APEEstimation.SectorBuilder_Tob_cff import *
from Alignment.APEEstimation.SectorBuilder_Tec_cff import *



###======================================================================================================================================================================

##
## One Sector for each Subdetector (means only one for e.g. both endcaps)
##

SubdetSectors = BPIX + FPIX + TIB + TOB + TID + TEC



###======================================================================================================================================================================

##
## Only TID and TEC (means only one for e.g. both endcaps)
##

TIDTEC = TID + TEC



###======================================================================================================================================================================

##
## Only TIB and TOB
##

TIBTOB = TIB + TOB



###======================================================================================================================================================================

##
## Only TIB and TOB, cosmic-like quartering (upper, lower, left, right part)
##

TIBTOBQuarters = TIBQuarters + TOBQuarters



###======================================================================================================================================================================

##
## Only TIB and TOB + Separation of pitches + Separation of 1D and 2D layers
##

TIBTOBPitchAnd2DSeparation = TIBPitchAnd2DSeparation + TOBPitchAnd2DSeparation



###======================================================================================================================================================================

##
## Only TIB and TOB, Separation of layers + rphi/stereo + orientations
##

# In TOB: All RPhi modules within a layer point in same w direction. Same is valid for Stereo modules, but with opposite sign

TIBTOBLayerAndOrientationSeparation = TIBLayerAndOrientationSeparation + TOBLayerAndOrientationSeparation


###======================================================================================================================================================================

##
## Only TID and TEC, Separation of side + rings + rphi/stereo
##

TIDTECSideAndRingSeparation = TIDSideAndRingSeparation + TECSideAndRingSeparation



###======================================================================================================================================================================

##
## Only TID and TEC, Separation of side + rings + rphi/stereo + orientations
##

# In TEC: All RPhi modules within a ring point in same w direction. Same is valid for Stereo modules, but with opposite sign

TIDTECSideAndRingAndOrientationSeparation = TIDSideAndRingAndOrientationSeparation + TECSideAndRingAndOrientationSeparation



###======================================================================================================================================================================

##
## Sectors used for validation
##

ValidationSectors = cms.VPSet(
    BpixLayer1Out,
    BpixLayer3In,
    FpixMinusLayer1,
    TibLayer1RphiOut,
    TibLayer4In,
    TobLayer1StereoOut,
    TobLayer5Out,
    TecPlusRing7,
)


###======================================================================================================================================================================

##
## Recent definition for whole tracker
##

RecentSectors = cms.VPSet()

RecentSectors += BPIXLayerAndOrientationSeparation
RecentSectors += FPIXSideAndLayerSeparation
RecentSectors += TIBLayerAndOrientationSeparation
RecentSectors += TOBLayerAndOrientationSeparation
RecentSectors += TIDSideAndRingSeparation
RecentSectors += TECSideAndRingSeparation








