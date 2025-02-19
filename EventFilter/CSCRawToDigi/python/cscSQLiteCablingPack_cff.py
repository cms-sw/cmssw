import FWCore.ParameterSet.Config as cms

# different es_sources are used for different purposes - packing and unpacking
# this one is for packing
cscPackingCabling = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCChamberMapRcd'),
        tag = cms.string('CSCChamberMap')
    )),
    connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/CSCChamberMapValues_18X.db')
)


