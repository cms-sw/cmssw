import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalFETriggerPrimitiveDigiProducer_cfi import *

hgcalTriggerPrimitives = cms.Sequence(hgcalFETriggerPrimitiveDigiProducer)
