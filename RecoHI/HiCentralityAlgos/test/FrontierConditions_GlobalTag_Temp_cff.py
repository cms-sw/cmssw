
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_Hydjet2760GeV_v1_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
             label = cms.untracked.string("")
             ),
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_PixelHits40_AMPT2760GeV_v3_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
             label = cms.untracked.string("PixelHitsAMPT_2760GeV")
             ),
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_Hydjet2760GeV_v1_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
             label = cms.untracked.string("HFhits")
             ),
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_AMPT2760GeV_v3_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
             label = cms.untracked.string("HFhitsAMPT_2760GeV")
             )
    )


