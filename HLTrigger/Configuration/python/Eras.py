import FWCore.ParameterSet.Config as cms

# import the relevant eras from Configuration.Eras.*

def modifyHLTforEras(fragment):
    """load all Eras-based customisations for the HLT configuration"""

    # modify the HLT configuration for the Phase I pixel geometry
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import modifyHLTforPhaseIPixelGeom
    modifyHLTforPhaseIPixelGeom(fragment)

    # modify the HLT configuration to run the Phase I tracking in the particle flow sequence
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import modifyHLTforPhaseIPFTracking
    modifyHLTforPhaseIPFTracking(fragment)
