import FWCore.ParameterSet.Config as cms

import Geometry.HcalEventSetup.hcalTopologyIdeal_cfi

hcalTopologyIdeal = Geometry.HcalEventSetup.hcalTopologyIdeal_cfi.hcalTopologyIdeal.clone()

import Geometry.HcalEventSetup.hcalTopologyConstants_cfi as hcalTopologyConstants_cfi
hcalTopologyIdeal.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
