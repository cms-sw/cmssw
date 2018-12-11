import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi import *
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import *


hgcalBackEndLayer2 = cms.Sequence(hgcalBackEndLayer2Producer)

