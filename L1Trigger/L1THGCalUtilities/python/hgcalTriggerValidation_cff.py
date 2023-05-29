import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cff import *

L1THGCalTriggerValidation = cms.Sequence(L1THGCalTriggerPrimitives*L1THGCalTriggerNtuples)


