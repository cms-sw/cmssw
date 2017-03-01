import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTriggerPrimitiveDigiProducer_cfi import *


hgcalTriggerPrimitives = cms.Sequence(hgcalTriggerPrimitiveDigiProducer)

hgcalTriggerPrimitives_reproduce = cms.Sequence(hgcalTriggerPrimitiveDigiFEReproducer)
