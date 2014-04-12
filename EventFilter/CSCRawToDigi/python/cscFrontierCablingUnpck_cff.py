import FWCore.ParameterSet.Config as cms

# different es_sources are used for different purposes - packing and unpacking
# this one is for unpacking
cscUnpackingCabling = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCCrateMapRcd'),
        tag = cms.string('CSCCrateMap')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_CSC') ##FrontierDev/CMS_COND_CSC"

)


