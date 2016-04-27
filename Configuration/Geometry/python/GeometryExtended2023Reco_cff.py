import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsExtendedGeometry2023simXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from Geometry.HcalCommonData.hcalParameters_cfi      import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cfi import *
from Geometry.HGCalCommonData.hgcalV6ParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalV6NumberingInitialization_cfi import *
