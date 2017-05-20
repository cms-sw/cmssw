import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_HI_cfi import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1caloparProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TCaloStage2ParamsRcd'),
                 tag = cms.string("L1TCaloParamsPrototype_Stage2v0_hlt")
            )
       )
)

L1TCaloParamsOnlineProd = cms.ESProducer("L1TCaloParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),

    layer1HOverE        = cms.string(''), #cms.string("layer1HOverE"),
    egBypassExtHOverE   = cms.string(''), #cms.string("egammaBypassExtHOverE"),
    egIsoLUT2           = cms.string(''), #cms.string("egammaIsoLUT2"),
    etSumBypassMetPUS   = cms.string(''), #cms.string("etSumBypassMetPUS"),
    etSumBypassEttPUS   = cms.string(''), #cms.string("etSumBypassEttPUS"),
    etSumBypassEcalSumPUS = cms.string(''), #cms.string("etSumBypassEcalSumPUS"),
    etSumMetPUSType     = cms.string(''), #cms.string("etSumMetPUSType"),
    etSumEttPUSType     = cms.string(''), #cms.string("etSumEttPUSType"),
    etSumEcalSumPUSType = cms.string(''), #cms.string("etSumEcalSumPUSType"),
    etSumMetPUSLUT      = cms.string(''), #cms.string("METX_energyCalibPUSLUT"),
    etSumEttPUSLUT      = cms.string(''), #cms.string("ET_energyCalibPUSLUT"),
    etSumEcalSumPUSLUT  = cms.string('') #cms.string("ecalET_energyCalibPUSLUT")
)

