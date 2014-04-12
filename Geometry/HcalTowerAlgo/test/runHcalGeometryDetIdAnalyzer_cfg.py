import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

process = cms.Process("HcalGeometryTest")

process.load("Configuration.Geometry.GeometryExtendedPostLS2_cff")
process.load("Geometry.HcalEventSetup.HcalTopology_cfi")

process.HcalHardcodeGeometryEP = cms.ESProducer( "HcalHardcodeGeometryEP" ,
                                                 appendToDataLabel = cms.string("_master"),
                                                 HcalReLabel = HcalReLabel
                                                 )
## Comment it out to test std Hcal geometry
##
import Geometry.HcalEventSetup.hcalSLHCTopologyConstants_cfi as hcalTopologyConstants_cfi
process.hcalTopologyIdeal.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
##

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hga = cms.EDAnalyzer("HcalGeometryDetIdAnalyzer",
                             HcalReLabel = HcalReLabel,
                             HCALGeometryLabel = cms.string("_master") )

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
