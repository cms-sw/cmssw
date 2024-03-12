# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_HCAL"
import FWCore.ParameterSet.Config as cms

#
# Hcal  calibrations from Frontier
#
#include "CalibCalorimetry/HcalPlugins/data/Hcal_FrontierConditions.cfi"
from CalibCalorimetry.HcalPlugins.Hcal_FrontierConditions_cff import *
es_pool.connect = 'frontier://FrontierDev/CMS_COND_HCAL'

# foo bar baz
# JVou6XyeVD4ar
# t6FKpW07B2m7H
