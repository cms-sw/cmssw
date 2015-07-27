import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsExtendedGeometryX0MaxXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from Geometry.HcalCommonData.hcalParameters_cfi      import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cfi import *

# Reconstruction geometry services
from Configuration.Geometry.GeometryReco_cff import *
