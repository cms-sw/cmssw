import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

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
#process.load("Configuration.Geometry.GeometryExtendedPostLS2_cff")
#process.load("Geometry.HcalEventSetup.HcalTopology_cfi")

## process.HcalHardcodeGeometryEP = cms.ESProducer( "HcalHardcodeGeometryEP" ,
##                                                  appendToDataLabel = cms.string("_master"),
##                                                  HcalReLabel = HcalReLabel
##                                                  )
## Comment it out to test std Hcal geometry
##
import Geometry.HcalEventSetup.hcalSLHCTopologyConstants_cfi as hcalTopologyConstants_cfi
process.hcalTopologyIdeal.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
##

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hga = cms.EDAnalyzer("HcalGeometryAnalyzer",
                             HcalReLabel = HcalReLabel,
                             HCALGeometryLabel = cms.string("") )

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
