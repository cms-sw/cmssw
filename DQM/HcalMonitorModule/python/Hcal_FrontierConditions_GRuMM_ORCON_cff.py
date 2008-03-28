import FWCore.ParameterSet.Config as cms

#include "CondCore/DBCommon/data/CondDBCommon.cfi"
#replace CondDBCommon.connect = "oracle://cms_orcoff_int2r/CMS_COND_HCAL"
#replace CondDBCommon.DBParameters.authenticationPath="/afs/cern.ch/cms/DB/conddb"
#replace CondDBCommon.timetype = "runnumber"
from CondCore.DBCommon.CondDBSetup_cfi import *
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

hcalConditions = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/nfshome0/hltpro/cmssw/cfg/')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('HcalPedestalsRcd'),
        tag = cms.string('hcal_pedestals_fC_v2_grmm')
    ), cms.PSet(
        record = cms.string('HcalPedestalWidthsRcd'),
        tag = cms.string('hcal_widths_fC_v2_grmm')
    ), cms.PSet(
        record = cms.string('HcalElectronicsMapRcd'),
        tag = cms.string('official_emap_v5_080208')
    ), cms.PSet(
        record = cms.string('HcalGainsRcd'),
        tag = cms.string('hcal_gains_v2_gren_reprocessing')
    ), cms.PSet(
        record = cms.string('HcalQIEDataRcd'),
        tag = cms.string('qie_normalmode_v3')
    )),
    connect = cms.string('frontier://(serverurl=http://frontier1.cms:8000/FrontierOn)(serverurl=http://frontier2.cms:8000/FrontierOn)(retrieve-ziplevel=0)/CMS_COND_ON_18X_HCAL'),
    siteLocalConfig = cms.untracked.bool(False)
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 'channelQuality')
)


