import FWCore.ParameterSet.Config as cms

# cfi for L1 GT Trigger Menu produced from an XML file
l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",
    # choose luminosity directory
    TriggerMenuLuminosity = cms.string('lumi1x1032'),
    # XML file for Global Trigger menu (def.xml) 
    DefXmlFile = cms.string('L1Menu2007.xml'),
    # XML file for Global Trigger VME configuration (vme.xml)                 
    VmeXmlFile = cms.string('')
)


