import FWCore.ParameterSet.Config as cms

#
# Hcal fake calibrations
#
#
# please note: in the future, it should load Hcal_FakeConditions.cfi from this same directory
# for 130 is was decided (by DPG) to stick to the old config, hence I load
#
#include "CalibCalorimetry/HcalPlugins/data/Hcal_FakeConditions.cfi"
from CalibCalorimetry.HcalPlugins.Hcal_FakeConditions_cff import *

