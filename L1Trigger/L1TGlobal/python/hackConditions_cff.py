#
# hackConditions for L1TGlobal package
#  

# Here are the conditions needed to configure the L1TGlobal package which are
# not yet in the Global Tag.  This differs from "fake conditions" because items
# will be removed from this list as the become available in the Global Tag.

import FWCore.ParameterSet.Config as cms


from L1Trigger.L1TGlobal.StableParametersConfig_cff import *

from L1Trigger.L1TGlobal.TriggerMenuXml_cfi import *
TriggerMenuXml.TriggerMenuLuminosity = 'startup'
#TriggerMenuXml.DefXmlFile = 'L1_Example_Menu_2013.xml'
#TriggerMenuXml.DefXmlFile = 'L1Menu_Reference_2014.xml'
#TriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2015_25nsStage1_v6_uGT_v2.xml'
TriggerMenuXml.DefXmlFile = 'L1Menu_Collisions2015_25nsStage1_v6_uGT_v3.xml'

from L1Trigger.L1TGlobal.TriggerMenuConfig_cff import *
es_prefer_l1GtParameters = cms.ESPrefer('l1t::TriggerMenuXmlProducer','TriggerMenuXml')

