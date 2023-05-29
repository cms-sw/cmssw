import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.l1tHGCalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalVFEProducer_cfi import *

L1THGCalVFE = cms.Task(l1tHGCalVFEProducer)
L1THFnoseVFE = cms.Task(l1tHFnoseVFEProducer)

