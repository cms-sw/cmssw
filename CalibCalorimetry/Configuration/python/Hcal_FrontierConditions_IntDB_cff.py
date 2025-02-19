# The following comments couldn't be translated into the new config version:

#cms_conditions_data/CMS_COND_20X_HCAL"
import FWCore.ParameterSet.Config as cms

#
# Hcal  calibrations from Frontier
#
#include "CalibCalorimetry/HcalPlugins/data/Hcal_FrontierConditions.cfi"
from CalibCalorimetry.HcalPlugins.Hcal_FrontierConditions_cff import *
es_pool.connect = 'frontier://cms_conditions_data/CMS_COND_20X_HCAL'

