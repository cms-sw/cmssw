#
# replace the L1 menu from the global tag with another menu
# see options in L1Trigger_custom.py
#
# V.M. Ghete 2010-06-09

import FWCore.ParameterSet.Config as cms

def customise(process):
    from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
    process = customiseL1Menu(process)
    return process


def L1Menu_Collisions2015_25ns_v0(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_25ns_v0_L1T_Scales_20101224_Imp0_0x102f.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_25ns_v1(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_25ns_v1_L1T_Scales_20101224_Imp0_0x102f.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_25ns_v2(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_25ns_v3(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_25nsStage1_v3_L1T_Scales_20141121_Imp0_0x1031.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_50ns_v0(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_50ns_v0_L1T_Scales_20141121_Imp0_0x1031.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_50ns_v1(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_50nsGct_v1_L1T_Scales_20141121_Imp0_0x1030.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_50ns_v2(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_50nsGct_v2_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_50ns_v3(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_50nsGct_v3_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_50ns_v4(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_50nsGct_v4_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_lowPU_v1(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_lowPU_v1_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_lowPU_v2(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_lowPU_v2_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_lowPU_v3(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_lowPU_v3_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_Collisions2015_lowPU_v4(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_Collisions2015_lowPU_v4_L1T_Scales_20141121.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer', 'l1GtTriggerMenuXml' )

    return process


def L1Menu_CollisionsHeavyIons2015_v0(process):
    process.load( 'L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi' )
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = 'startup'
    process.l1GtTriggerMenuXml.DefXmlFile            = 'L1Menu_CollisionsHeavyIons2011_v0_nobsc_notau_centrality_q2_singletrack.v1.xml'

    process.load( 'L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff' )
    process.es_prefer_l1GtParameters = cms.ESPrefer( 'L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml' )

    return process

