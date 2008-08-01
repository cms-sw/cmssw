import FWCore.ParameterSet.Config as cms

# cff file for L1Menu2008_2E30 - prescaled version
# to be added after L1GtConfig.cff
# menu definition
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *
# prescale factors for algorithm triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_PrescaleFactorsAlgoTrig_cff import *
# trigger mask for algorithm trigger
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_TriggerMaskAlgoTrig_cff import *
l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1030'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu2008_2E30.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

