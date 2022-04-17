import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import *


hgcalBackEndLayer1 = cms.Task(hgcalBackEndLayer1Producer)
hgcalBackEndStage1 = cms.Task(hgcalBackEndStage1Producer)

hgcalBackEndLayer1HFNose = cms.Task(hgcalBackEndLayer1ProducerHFNose)
