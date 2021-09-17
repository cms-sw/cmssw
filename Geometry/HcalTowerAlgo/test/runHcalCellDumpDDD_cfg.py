import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalGeometryTest")

process.load("Geometry.CMSCommonData.cmsExtendedGeometry2021XML_cfi")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.HcalCommonData.hcalDDDRecConstants_cfi")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")
process.load("Geometry.HcalTowerAlgo.hcalCellParameterDump_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.hcalCellParameterDump)
