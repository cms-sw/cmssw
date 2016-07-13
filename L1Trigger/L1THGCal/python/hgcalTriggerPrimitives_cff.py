import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerPrimitiveDigiProducer_cfi import *

hgcalTriggerPrimitives = cms.Sequence(hgcalTriggerPrimitiveDigiProducer)
