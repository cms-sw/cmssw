import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage2Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone()

import CondCore.ESSources.CondDBESSource_cfi
es_pool_hf1x1 = CondCore.ESSources.CondDBESSource_cfi.GlobalTag.clone()

es_pool_hf1x1.timetype = cms.string('runnumber')
es_pool_hf1x1.toGet = cms.VPSet(
            cms.PSet(record = cms.string("HcalLutMetadataRcd"),
                     tag = cms.string("HcalLutMetadata_HFTP_1x1")
                     ),
            cms.PSet(record = cms.string("HcalElectronicsMapRcd"),
                     tag = cms.string("HcalElectronicsMap_HFTP_1x1")
                     )
            )
es_pool_hf1x1.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
es_pool_hf1x1.authenticationMethod = cms.untracked.uint32(0)

es_prefer_es_pool_hf1x1 = cms.ESPrefer("PoolDBESSource", "es_pool_hf1x1")    

#def L1TEventSetupForHF1x1TPs(process):
#    process.es_pool_hf1x1 = cms.ESSource(
#        "PoolDBESSource",
#        #process.CondDBSetup,
#        timetype = cms.string('runnumber'),
#        toGet = cms.VPSet(
#            cms.PSet(record = cms.string("HcalLutMetadataRcd"),
#                     tag = cms.string("HcalLutMetadata_HFTP_1x1")
#                     ),
#            cms.PSet(record = cms.string("HcalElectronicsMapRcd"),
#                     tag = cms.string("HcalElectronicsMap_HFTP_1x1")
#                     )
#            ),
#        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
#        authenticationMethod = cms.untracked.uint32(0)
#        )
#    process.es_prefer_es_pool_hf1x1 = cms.ESPrefer("PoolDBESSource", "es_pool_hf1x1")    
