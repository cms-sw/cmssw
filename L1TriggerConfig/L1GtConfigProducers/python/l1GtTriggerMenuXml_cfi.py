import FWCore.ParameterSet.Config as cms

# cfi for L1 GT Trigger Menu produced from an XML file

l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",

    # choose luminosity directory
    TriggerMenuLuminosity = cms.string('lumi1030'),
    
    # XML file for Global Trigger menu (def.xml) 
    DefXmlFile = cms.string('L1Menu_2008MC_2E30.xml'),
    
    # XML file for Global Trigger VME configuration (vme.xml)                 
    VmeXmlFile = cms.string('')
)


