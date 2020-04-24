import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsExtendedGeometry2015MuonGEMDevXML_cfi import *
#from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
#from Geometry.HcalCommonData.hcalParameters_cfi      import *
#from Geometry.HcalCommonData.hcalDDDSimConstants_cfi import *

from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometryDB_cfi import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cfi import *

