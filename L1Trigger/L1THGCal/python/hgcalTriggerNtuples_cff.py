import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerNtuples_cfi import *

hgcalTriggerNtuples = cms.Sequence(hgcalTriggerNtuplizer)
