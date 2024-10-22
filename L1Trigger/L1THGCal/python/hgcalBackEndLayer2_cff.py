import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.l1tHGCalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer2Producer_cfi import *


L1THGCalBackEndLayer2 = cms.Task(l1tHGCalBackEndLayer2Producer)
L1THGCalBackEndStage2 = cms.Task(l1tHGCalBackEndStage2Producer)

L1THGCalBackEndLayer2HFNose = cms.Task(l1tHGCalBackEndLayer2ProducerHFNose)
