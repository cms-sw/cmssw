import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalGeometryTest")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']
process.XMLFromDBSource.label=''
 
process.GlobalTag.toGet = cms.VPSet(
         cms.PSet(record = cms.string("GeometryFileRcd"),
                  tag = cms.string("XMLFILE_Geometry_61YV5_Phase1_R30F12_HCal_Ideal_mc"),
                  connect = cms.untracked.string("sqlite_file:./myfile.db")
                  ),
         cms.PSet(record = cms.string("PHcalRcd"),
                  tag = cms.string("HCALRECO_Geometry_61YV5"),
                  connect = cms.untracked.string("sqlite_file:./myfile.db")
                  )
         )

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hga = cms.EDAnalyzer("HcalGeometryAnalyzer",
                             UseOldLoader   = cms.bool(False),
                             GeometryFromDB = cms.bool(True))

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
