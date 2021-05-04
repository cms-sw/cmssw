import FWCore.ParameterSet.Config as cms

## Fills CondFormats from the database
from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")


## Fills firmware, pT LUT, and PC LUT versions from the database
emtfParamsSource = cms.ESSource(
    "PoolDBESSource",
    CondDB,
    toGet   = cms.VPSet(
        cms.PSet(
            record = cms.string("L1TMuonEndCapParamsRcd"),
            tag    = cms.string("L1TMuonEndCapParams_Stage2v3_hlt")
            )
        )
    )


## Fills pT LUT XMLs ("forests") from the database
emtfForestsDB = cms.ESSource(
    "PoolDBESSource",
    CondDB,
    toGet   = cms.VPSet(
        cms.PSet(
            ## https://cms-conddb.cern.ch/cmsDbBrowser/search/Prod/L1TMuonEndCapForest
            record = cms.string("L1TMuonEndCapForestRcd"),
            ## v7 EMTF pT LUTs from June 8, 2017
            tag = cms.string("L1TMuonEndCapForest_Stage2v1_2018_HI_mc")
            )
        )
    )
