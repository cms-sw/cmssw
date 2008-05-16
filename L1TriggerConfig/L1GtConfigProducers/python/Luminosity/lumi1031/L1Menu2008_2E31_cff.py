import FWCore.ParameterSet.Config as cms

# prescale factors for algorithm triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_PrescaleFactorsAlgoTrig_cff import *
# trigger mask for algorithm trigger
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_TriggerMaskAlgoTrig_cff import *
# cff file for L1Menu2008_2E31 - prescaled version
# to be added after L1GtConfig.cff
# menu definition
l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1031'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu2008_2E31.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

