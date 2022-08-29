import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import *

L1THGCalVFE = cms.Task(l1tHGCalVFEProducer)
L1THFnoseVFE = cms.Task(l1tHFnoseVFEProducer)

