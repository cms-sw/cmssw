import warnings
warnings.warn('ATTENTION: This job uses Run 1 Geometry.')

import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# TrackerAdditionalParametersPerDet contains only default values, needed for consistency with Phase 2
from Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *

