import FWCore.ParameterSet.Config as cms

# cfi for L1 GT Trigger Menu produced from an XML file

l1uGtTriggerMenuXml = cms.ESProducer("l1t::L1uGtTriggerMenuXmlProducer",

    # choose luminosity directory
    TriggerMenuLuminosity = cms.string('startup'),
    
    # XML file for Global Trigger menu (def.xml) 
    DefXmlFile = cms.string('L1Menu_Commissioning2009_v1_L1T_Scales_20080926_startup_Imp0.xml'),
    
    # XML file for Global Trigger VME configuration (vme.xml)                 
    VmeXmlFile = cms.string('')
)


