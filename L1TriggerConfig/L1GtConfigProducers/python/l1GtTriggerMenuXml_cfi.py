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
eras.run2_25ns_specific.toModify( l1GtTriggerMenuXml, DefXmlFile = 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030.xml' )

## The following two menus haven't been implemented yet, so until then use
## the same menu as for 25ns.
#eras.run2_50ns_specific.toModify( l1GtTriggerMenuXml, DefXmlFile = 'L1Menu_Collisions2015_50ns_v0_L1T_Scales_20141121_Imp0_0x1031.xml' )
#eras.run2_HI_specific.toModify( l1GtTriggerMenuXml,   DefXmlFile = 'L1Menu_CollisionsHeavyIons2011_v0_nobsc_notau_centrality_q2_singletrack.v1.xml' )
## FIXME - take out these lines and uncomment the ones above once the menus are implemented.
eras.run2_50ns_specific.toModify( l1GtTriggerMenuXml, DefXmlFile = 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030.xml' )
eras.run2_HI_specific.toModify( l1GtTriggerMenuXml,   DefXmlFile = 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030.xml' )
