import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

def modifyHLTforEras(fragment):
    """load all Eras-based customisations for the HLT configuration"""

    # modify the HLT configuration for the Phase I pixel geometry
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import customizeHLTPhaseIPixelGeom
    eras.trackingPhase1.toModify(fragment, customizeHLTPhaseIPixelGeom)

    # modify the HLT configuration to run the Phase I tracking in the particle flow sequence
    from HLTrigger.Configuration.customizeHLTTrackingForPhaseI2017 import customizeHLTForPFTrackingPhaseI2017
    eras.trackingPhase1.toModify(fragment, customizeHLTForPFTrackingPhaseI2017)
