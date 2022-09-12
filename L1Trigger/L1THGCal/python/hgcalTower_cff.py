import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import *


L1THGCalTower = cms.Task(l1tHGCalTowerProducer)

L1THGCalTowerHFNose = cms.Task(l1tHGCalTowerProducerHFNose)

