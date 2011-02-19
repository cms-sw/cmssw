
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_PixelHits40_Hydjet2760GeV_v0_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label = cms.untracked.string("PixelHitsHydjet_2760GeV")
             ),
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_PixelHits40_AMPT2760GeV_v0_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label = cms.untracked.string("PixelHitsAMPT_2760GeV")
             ),
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_Hydjet2760GeV_v0_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label = cms.untracked.string("HFhitsHydjet_2760GeV")
             ),
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_AMPT2760GeV_v0_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label = cms.untracked.string("HFhitsAMPT_2760GeV")
             )
    )


