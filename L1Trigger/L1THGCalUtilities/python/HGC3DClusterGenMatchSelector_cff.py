import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.HGC3DClusterGenMatchSelector_cfi import *

L1THGCalTriggerSelector = cms.Sequence(l1tHGCal3DClusterGenMatchSelector)
