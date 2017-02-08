import FWCore.ParameterSet.Config as cms

def modifyHLTforEras(fragment):
    """load all Eras-based customisations for the HLT configuration"""

    # modify the HLT configuration for the Phase I pixel geometry
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import modifyHLTPhaseIPixelGeom
    modifyHLTPhaseIPixelGeom(fragment)

    # modify the HLT configuration to run the Phase I tracking in the particle flow sequence
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import modifyHLTForPFTrackingPhaseI2017
    modifyHLTForPFTrackingPhaseI2017(fragment)

    # modify the HLT configuration for the Phase I HE upgrade
    from HLTrigger.Configuration.customizeHLTforHCALPhaseI import modifyHLTforHEforPhaseI
    modifyHLTforHEforPhaseI(fragment)

    # modify the HLT configuration for the Phase I HF upgrade
    from HLTrigger.Configuration.customizeHLTforHCALPhaseI import modifyHLTforHFforPhaseI
    modifyHLTforHFforPhaseI(fragment)

