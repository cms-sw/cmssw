import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.l1tHGCalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import *


L1THGCalBackEndLayer1 = cms.Task(l1tHGCalBackEndLayer1Producer)
L1THGCalBackEndStage1 = cms.Task(l1tHGCalBackEndStage1Producer)

L1THGCalBackEndLayer1HFNose = cms.Task(l1tHGCalBackEndLayer1ProducerHFNose)
