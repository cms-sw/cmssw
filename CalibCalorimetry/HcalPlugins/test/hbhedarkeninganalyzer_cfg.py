import FWCore.ParameterSet.Config as cms

process = cms.Process("demo")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryExtended2018_cff")
#process.load("Configuration.Geometry.GeometryExtended2023D4_cff")
process.load("Geometry.HcalCommonData.hcalParameters_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cfi")
process.load("Geometry.HcalCommonData.hcalDDDRecConstants_cfi")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

from DataFormats.HcalCalibObjects.HBHEDarkeningParameters_cff import *

process.ana = cms.EDAnalyzer("HBHEDarkeningAnalyzer",
    deliveredLumi = cms.double(304),
    HBDarkeningParameters = HBDarkeningParameters,
    HEDarkeningParameters = HEDarkeningParameters,
)

process.p1 = cms.Path(process.ana)
