import FWCore.ParameterSet.Config as cms

# cff file for L1Menu2007 - prescaled version
# to be added after L1GtConfig_cff

# menu definition
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *

l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1x1032'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu2007.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

# prescale factors for algorithm triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_PrescaleFactorsAlgoTrig_cff import *

# trigger mask for algorithm trigger
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_TriggerMaskAlgoTrig_cff import *

# trigger veto mask for algorithm triggers - NOT AVAILABLE IN HARDWARE: DO NOT USE IT
# default: no bit vetoed 

# prescale factors, trigger mask, trigger veto mask for technical triggers
# default: no prescale, no bit masked, no bit vetoed 

# foo bar baz
# Bz2QlTYsFTSKh
# 7YPykkM3BzAST
