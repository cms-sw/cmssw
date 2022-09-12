import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.l1tHGCalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalTowerProducer_cfi import *


L1THGCalTower = cms.Task(l1tHGCalTowerProducer)

L1THGCalTowerHFNose = cms.Task(l1tHGCalTowerProducerHFNose)

