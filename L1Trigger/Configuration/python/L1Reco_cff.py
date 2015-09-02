import FWCore.ParameterSet.Config as cms

# L1 reconstruction sequence for data and MC
#     L1Extra (all BxInEvent)
#     L1GtTriggerMenuLite
#     l1GtRecord - requires functional L1 O2O
#     l1L1GtObjectMap - L1 trigger object maps, now produced with the convertor
#     
# V.M. Ghete 2009-07-11

 
# L1Extra
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
l1extraParticles.centralBxOnly = False

# L1 GT lite record
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *

# L1GtTriggerMenuLite
from EventFilter.L1GlobalTriggerRawToDigi.l1GtTriggerMenuLite_cfi import *

#
# If the Stage 1 trigger is running, there is also some different configuration than
# the general Run 2 stuff.
#
def _customiseForStage1( processObject ) :
    processObject.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')
    processObject.load('L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi')
# A unique name is required so I'll use "modify<python filename>ForStage1Trigger_"
from Configuration.StandardSequences.Eras import eras
modifyL1TriggerConfigurationL1RecoForStage1Trigger_ = eras.stage1L1Trigger.makeProcessModifier( _customiseForStage1 )

# conditions in edm
import EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi
conditionsInEdm = EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi.conditionDumperInEdm.clone()

# l1L1GtObjectMap
from L1Trigger.GlobalTrigger.convertObjectMapRecord_cfi import *
l1L1GtObjectMap = convertObjectMapRecord.clone()

# sequences

L1Reco_L1Extra = cms.Sequence(l1extraParticles)
L1Reco_L1Extra_L1GtRecord = cms.Sequence(l1extraParticles+l1GtRecord)
#
L1Reco = cms.Sequence(l1extraParticles+l1GtTriggerMenuLite+conditionsInEdm+l1L1GtObjectMap)
