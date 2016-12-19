import FWCore.ParameterSet.Config as cms

# import the relevant eras from Configuration.Eras.*
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

def modifyHLTforEras(fragment):
    """load all Eras-based customisations for the HLT configuration"""

    # modify the HLT configuration for the Phase I pixel geometry
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import modifyHLTPhaseIPixelGeom
    modifyHLTPhaseIPixelGeom(fragment)

    # modify the HLT configuration to run the Phase I tracking in the particle flow sequence
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import modifyHLTForPFTrackingPhaseI2017
    modifyHLTForPFTrackingPhaseI2017(fragment)
