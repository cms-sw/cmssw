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

#
# Make changes for Run 2
#
from Configuration.StandardSequences.Eras import eras
# Change the trigger menu depending on whether this is 25ns, 50ns or HI running
eras.run2_50ns_specific.toModify( l1GtTriggerMenuXml, DefXmlFile = 'L1Menu_Collisions2015_50nsGct_v1_L1T_Scales_20141121_Imp0_0x1030.xml' )
eras.run2_25ns_specific.toModify( l1GtTriggerMenuXml, DefXmlFile = 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030.xml' )
eras.run2_HI_specific.toModify( l1GtTriggerMenuXml,   DefXmlFile = 'L1Menu_CollisionsHeavyIons2011_v0_nobsc_notau_centrality_q2_singletrack.v1.xml' )
