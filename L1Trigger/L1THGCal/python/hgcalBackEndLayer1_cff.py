import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import *


L1THGCalBackEndLayer1 = cms.Task(l1tHGCalBackEndLayer1Producer)
L1THGCalBackEndStage1 = cms.Task(l1tHGCalBackEndStage1Producer)

L1THGCalBackEndLayer1HFNose = cms.Task(l1tHGCalBackEndLayer1ProducerHFNose)
