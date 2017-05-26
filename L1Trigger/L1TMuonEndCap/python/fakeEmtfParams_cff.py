import FWCore.ParameterSet.Config as cms

emtfParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonEndcapParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

## EMTF ESProducer. Fills CondFormats from XML files.
emtfParams = cms.ESProducer(
    "L1TMuonEndCapParamsESProducer",
    PtAssignVersion = cms.int32(1),
    firmwareVersion = cms.int32(12345),
    ## Not yet implemented in O2O
    ## v1 corresponds to data/emtf_luts/ph_lut_v1, used for all of 2016
    ## v2 corresponds to data/emtf_luts/ph_lut_v2, used at the beginning of 2017
    PrimConvVersion = cms.int32(2)
)


emtfForestsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonEndCapForestRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")

emtfForestsDB = cms.ESSource(
    "PoolDBESSource",
    CondDB,
    toGet   = cms.VPSet(
        cms.PSet(
            ## https://cms-conddb.cern.ch/cmsDbBrowser/search/Prod/L1TMuonEndCapForest
            record = cms.string("L1TMuonEndCapForestRcd"),
            ## v6 EMTF pT LUTs from May 24, 2017
            tag = cms.string("L1TMuonEndCapForest_static_Sq_20170523_mc")
            )
        )
    )




