import FWCore.ParameterSet.Config as cms

# cff file for L1Menu_CRUZET200805, key gr7_muon_hcal
# to be added after L1GtConfig_cff

# menu definition
from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *

l1GtTriggerMenuXml.TriggerMenuLuminosity = 'lumi1x1032'
l1GtTriggerMenuXml.DefXmlFile = 'L1Menu_CRUZET200805.xml'
l1GtTriggerMenuXml.VmeXmlFile = ''

# prescale factors for algorithm triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_CRUZET200805_PrescaleFactorsAlgoTrig_cff import *

# trigger mask for algorithm trigger
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_CRUZET200805_TriggerMaskAlgoTrig_gr7_muon_hcal_cff import *

# prescale factors, trigger veto mask for technical triggers
# default: no prescale, no bit masked, no bit vetoed 

# trigger mask for technical triggers
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu_CRUZET200805_TriggerMaskTechTrig_cff import *

