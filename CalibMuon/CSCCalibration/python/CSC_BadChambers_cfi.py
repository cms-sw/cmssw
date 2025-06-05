import FWCore.ParameterSet.Config as cms

cscBadChambers = cms.ESSource("PoolDBESSource",
   DBParameters = cms.PSet(
       authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
   ),
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('CSCBadChambersRcd'),
       tag = cms.string('CSCBadChambers_empty_mc')
   )),
   connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/BadChambers_empty_mc.db')
)


