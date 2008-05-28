import FWCore.ParameterSet.Config as cms

# cff file for L1Menu2007 - unprescaled version 
# (prescale factors set to 1, except MinBias and ZeroBias) 
# to be added after L1GtConfig.cff
# menu definition
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *
# prescale factors for algorithm triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_PrescaleFactorsAlgoTrigUnprescale_cff import *
# trigger mask for algorithm trigger
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_TriggerMaskAlgoTrig_cff import *
l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1x1032'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu2007.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

