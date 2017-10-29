import FWCore.ParameterSet.Config as cms

## Fills CondFormats from the database
from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")


## Fills firmware, pT LUT, and PC LUT versions manually 
emtfParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonEndCapParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
    )

emtfParams = cms.ESProducer(
    "L1TMuonEndCapParamsESProducer",
    ## Version 5 was used for all of 2016
    PtAssignVersion = cms.int32(5),
    ## Latest version in 2016
    FirmwareVersion = cms.int32(49999),
    ## v0 corresponds to data/emtf_luts/ph_lut_v1, used for all of 2016
    PrimConvVersion = cms.int32(0)
    )


## Fills pT LUT XMLs ("forests") from the database
emtfForestsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonEndCapForestRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
    )

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
