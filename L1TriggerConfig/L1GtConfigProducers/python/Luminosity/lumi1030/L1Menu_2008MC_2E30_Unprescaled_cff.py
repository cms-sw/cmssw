import FWCore.ParameterSet.Config as cms

# cff file for L1Menu_2008MC_2E30 - unprescaled version 
# (prescale factors set to 1) 

# to be added after L1GtConfig_cff

# menu definition
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *

l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1030'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu_2008MC_2E30.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

# prescale factors for algorithm triggers
# default: no algorithm prescaled

# trigger mask for algorithm trigger
# default: no algorithm masked

# trigger veto mask for algorithm triggers - NOT AVAILABLE IN HARDWARE: DO NOT USE IT
# default: no bit vetoed 

# prescale factors, trigger mask, trigger veto mask for technical triggers
# default: no prescale, no bit masked, no bit vetoed 
