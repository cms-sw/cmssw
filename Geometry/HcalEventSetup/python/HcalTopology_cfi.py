import FWCore.ParameterSet.Config as cms

from Geometry.HcalCommonData.hcalRecNumberingInitialization_cfi import *

import Geometry.HcalEventSetup.hcalTopologyIdeal_cfi

hcalTopologyIdeal = Geometry.HcalEventSetup.hcalTopologyIdeal_cfi.hcalTopologyIdeal.clone()
