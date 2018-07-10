import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalVFEProducer_cfi import *
from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import *
from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import *


hgcalTriggerPrimitives = cms.Sequence(hgcalVFEProducer*hgcalConcentratorProducer*hgcalBackEndLayer1Producer*hgcalBackEndLayer2Producer*hgcalTowerMapProducer*hgcalTowerProducer)
