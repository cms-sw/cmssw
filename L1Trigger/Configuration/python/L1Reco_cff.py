import FWCore.ParameterSet.Config as cms

# L1 reconstruction sequence for data and MC
#     L1Extra (all BxInEvent)
#     L1GtTriggerMenuLite
#     l1GtRecord - requires functional L1 O2O
#     
# V.M. Ghete 2009-07-11

 
# L1Extra
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
l1extraParticles.centralBxOnly = False

# L1 GT lite record
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *

# L1GtTriggerMenuLite
from EventFilter.L1GlobalTriggerRawToDigi.l1GtTriggerMenuLite_cfi import *

# conditions in edm
import EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi
conditionsInEdm = EventFilter.L1GlobalTriggerRawToDigi.ConditionDumperInEdm.conditionDumperInEdm.clone()

# sequences

L1Reco_L1Extra = cms.Sequence(l1extraParticles)
L1Reco_L1Extra_L1GtRecord = cms.Sequence(l1extraParticles+l1GtRecord)
#
L1Reco = cms.Sequence(l1extraParticles+l1GtTriggerMenuLite+conditionsInEdm)
