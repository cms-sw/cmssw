import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cfi import *

hgcalTriggerNtuples = cms.Sequence(hgcalTriggerNtuplizer)


