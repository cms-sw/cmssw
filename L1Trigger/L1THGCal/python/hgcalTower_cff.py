import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import *


hgcalTower = cms.Sequence(hgcalTowerProducer)

