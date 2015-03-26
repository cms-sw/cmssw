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
# If this is Run 2 then the trigger menu needs to be loaded from XML
# instead of using l1GtTriggerMenuLite
#
def _loadMenuFromXML( processObject ) :
    """
    Imports the required producer to load the L1 menu from XML and creates an
    ESPrefer to prefer that. The actual filename of the menu is specified in
    era customisations within these imports.
    """
    processObject.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    processObject.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    processObject.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )
from Configuration.StandardSequences.Eras import eras
# A unique name is required for this object, so I'll call it "modify<python filename>ForRun2_"
modifyL1TriggerConfigurationL1RecoForRun2_ = eras.run2_common.makeProcessModifier( _loadMenuFromXML )

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
