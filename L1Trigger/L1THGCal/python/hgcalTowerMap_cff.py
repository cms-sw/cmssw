import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import *


L1THGCalTowerMap = cms.Task(l1tHGCalTowerMapProducer)

L1THGCalTowerMapHFNose = cms.Task(l1tHGCalTowerMapProducerHFNose)

