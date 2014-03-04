import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsExtendedGeometryInflated10TIBFlangeXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from Geometry.HcalCommonData.hcalSimNumberingInitialization_cfi import *

# Reconstruction geometry services
from Configuration.Geometry.GeometryReco_cff import *
