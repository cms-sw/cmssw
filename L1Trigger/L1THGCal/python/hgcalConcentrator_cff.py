import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import *


L1THGCalConcentrator = cms.Task(l1tHGCalConcentratorProducer)
L1THGCalConcentratorHFNose = cms.Task(l1tHGCalConcentratorProducerHFNose)

