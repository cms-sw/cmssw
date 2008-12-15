import FWCore.ParameterSet.Config as cms

#Include configuration ParameterSets
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigParams_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigMap_cff import *
L1DTConfigFromDB = cms.ESProducer("DTConfigDBProducer",
    DTTPGMapBlock,
    DTTPGParametersBlock,
    cfgConfig  = cms.bool(False),
    debugDB    = cms.bool(False),
    debugBti   = cms.int32(0),
    debugTraco = cms.int32(0),
    debugTSP   = cms.bool(False),
    debugTST   = cms.bool(False),
    debugTU    = cms.bool(False),
    debugSC    = cms.bool(False),
    debugLUTs  = cms.bool(False),             
    debug      = cms.bool(False),
    catalog = cms.string("file:testcatalog.xml"),
    authPath = cms.string("/afs/cern.ch/cms/DB/conddb"),
    siteLocalConfig = cms.bool(False),
    contact = cms.string("oracle://cms_orcoff_prod/CMS_COND_21X_DT"),
    tag = cms.string("conf_ccb_V01"),
    token = cms.string("[DB=00000000-0000-0000-0000-000000000000][CNT=DTConfigList][CLID=9CB14BE8-30A2-DB11-9935-000E0C5CE283][TECH=00000B01][OID=00000004-00000000]")
)



