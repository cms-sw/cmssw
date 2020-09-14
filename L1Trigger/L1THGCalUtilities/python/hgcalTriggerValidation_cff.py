import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cff import *

hgcalTriggerValidation = cms.Sequence(hgcalTriggerPrimitives*hgcalTriggerNtuples)


