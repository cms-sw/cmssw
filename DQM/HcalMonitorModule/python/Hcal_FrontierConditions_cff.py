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

#es_source es_pool = PoolDBESSource { 
#      using CondDBSetup
#      string connect = "frontier://FrontierDev/CMS_COND_HCAL"
#      string timetype = "runnumber"    
#      untracked uint32 authenticationMethod = 0
#           VPSet toGet = {
#                    {string record = "HcalPedestalsRcd"
#                     string tag    = "hcal_pedestals_fC_v1_zdc"
#                    },
#                    {string record = "HcalPedestalWidthsRcd"
#                     string tag =    "hcal_widths_fC_v1_zdc"
#                    },
#                    {string record = "HcalGainsRcd"
#                     string tag =    "hcal_gains_v1_zdc"
#                    },
#                    {string record = "HcalQIEDataRcd"
#                     string tag =    "qie_normalmode_v3_zdc"
#                    },
#                    {string record = "HcalElectronicsMapRcd"
#                     string tag =    "official_emap_16x_v1"
#                    }
# this is w/o zdc:
#                    {string record = "HcalPedestalsRcd"
#                     string tag    = "hcal_pedestals_fC_v1"
#                    },
#                    {string record = "HcalPedestalWidthsRcd"
#                     string tag =    "hcal_widths_fC_v1"
#                    },
#                    {string record = "HcalGainsRcd"
#                     string tag =    "hcal_gains_v1"
#                    },
#                    {string record = "HcalQIEDataRcd"
#                     string tag =    "qie_normalmode_v3"
#                    },
# this was for 15x series
#                    {string record = "HcalElectronicsMapRcd"
#                     string tag =    "emapTest15"
#                    }
#                  }
#             }
hcalConditions = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/nfshome0/hltpro/cmssw/cfg/')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('HcalPedestalsRcd'),
        tag = cms.string('hcal_pedestals_fC_gren')
    ), cms.PSet(
        record = cms.string('HcalElectronicsMapRcd'),
        tag = cms.string('official_emap_16x_v2')
    ), cms.PSet(
        record = cms.string('HcalGainsRcd'),
        tag = cms.string('hcal_gains_v1')
    ), cms.PSet(
        record = cms.string('HcalQIEDataRcd'),
        tag = cms.string('qie_normalmode_v3')
    ), cms.PSet(
        record = cms.string('HcalPedestalWidthsRcd'),
        tag = cms.string('hcal_widths_fC_gren')
    )),
    connect = cms.string('frontier://(serverurl=http://frontier1.cms:8000/FrontierOn)(serverurl=http://frontier2.cms:8000/FrontierOn)(retrieve-ziplevel=0)/CMS_COND_ON_170_HCAL'),
    siteLocalConfig = cms.untracked.bool(False)
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 'channelQuality')
)


