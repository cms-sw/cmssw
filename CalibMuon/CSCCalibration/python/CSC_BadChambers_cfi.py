import FWCore.ParameterSet.Config as cms

cscBadChambers = cms.ESSource("PoolDBESSource",
   DBParameters = cms.PSet(
       authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
       authenticationMethod = cms.untracked.uint32(1)
   ),
   timetype = cms.string('runnumber'),
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('CSCBadChambersRcd'),
       tag = cms.string('CSCBadChambers_empty_mc')
   )),
   connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/BadChambers_empty_mc.db')
)


