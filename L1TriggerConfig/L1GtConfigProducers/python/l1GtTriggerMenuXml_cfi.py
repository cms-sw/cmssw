import FWCore.ParameterSet.Config as cms

# cfi for L1 GT Trigger Menu produced from an XML file

l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",

    # choose luminosity directory
    TriggerMenuLuminosity = cms.string('startup'),
    
    # XML file for Global Trigger menu (def.xml) 
    DefXmlFile = cms.string('L1Menu_Commissioning2009_v1_L1T_Scales_20080926_startup_Imp0.xml'),
    
    # XML file for Global Trigger VME configuration (vme.xml)                 
    VmeXmlFile = cms.string('')
)

##
## Make changes for Run 2
##
from Configuration.StandardSequences.Eras import eras
eras.run2.toModify( l1GtTriggerMenuXml, DefXmlFile = 'L1Menu_Collisions2015_25ns_v1_L1T_Scales_20101224_Imp0_0x102f.xml' )
