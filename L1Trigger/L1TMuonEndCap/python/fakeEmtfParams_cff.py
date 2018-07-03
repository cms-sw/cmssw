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
    ## Version 6 is the first version used in 2017
    ## Version 7 was deployed June 8, 2017
    PtAssignVersion = cms.int32(7),
    ## 123456 is default (most up-to-date) firmware version
    ## Versions < 50000 correspond to 2016 (labeled according to Alex's FW version number)
    ## Versions > 50000 coorespond to 2017 (labeled automatically by timestamp)
    FirmwareVersion = cms.int32(123456),
    ## v0 corresponds to data/emtf_luts/ph_lut_v1, used for all of 2016
    ## v1 corresponds to data/emtf_luts/ph_lut_v2, used at the beginning of 2017
    PrimConvVersion = cms.int32(1)
    )


## Fills pT LUT XMLs ("forests") from the database
emtfForestsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonEndCapForestRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
    )

#emtfForestsDB = cms.ESSource(
#    "PoolDBESSource",
#    CondDB,
#    toGet   = cms.VPSet(
#        cms.PSet(
#            ## https://cms-conddb.cern.ch/cmsDbBrowser/search/Prod/L1TMuonEndCapForest
#            record = cms.string("L1TMuonEndCapForestRcd"),
#
#            # ## v5 EMTF pT LUTs from ~August 2016
#            # tag = cms.string("L1TMuonEndCapForest_static_2016_mc")
#            # ## v6 EMTF pT LUTs from May 24, 2017
#            # tag = cms.string("L1TMuonEndCapForest_static_Sq_20170523_mc")
#            ## v7 EMTF pT LUTs from June 7, 2017 - AWB 07.06.17
#            tag = cms.string("L1TMuonEndCapForest_static_Sq_20170613_v7_mc")
#            )
#        )
#    )

# ## EMTF ESProducer. Fills CondFormats from local XML files instead of database.
# emtfForests = cms.ESProducer(
#     "L1TMuonEndCapForestESProducer",
#     PtAssignVersion = cms.int32(7),
#     bdtXMLDir = cms.string("2017_v7")
#     )
