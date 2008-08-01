import FWCore.ParameterSet.Config as cms

# cff file for L1Menu2008_2E31 - unprescaled version 
# (prescale factors set to 1) 
# to be added after L1GtConfig.cff
# menu definition
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *
# prescale factors for algorithm triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_PrescaleFactorsAlgoTrigUnprescale_cff import *
# trigger mask for algorithm trigger
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_TriggerMaskAlgoTrig_cff import *
l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1031'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu2008_2E31.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

