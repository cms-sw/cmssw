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
            ## This payload contains
            ## PtAssignVersion=5, firmwareVersion=49999, PhiMatchWindowSt1=0
            tag    = cms.string("L1TMuonEndCapParams_static_2016_mc")
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
            ## v5 EMTF pT LUTs from ~August 2016
            tag = cms.string("L1TMuonEndCapForest_static_2016_mc")
            )
        )
    )
