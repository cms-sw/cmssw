import FWCore.ParameterSet.Config as cms

process = cms.Process("HcalGeometryTest")

process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
process.load("Geometry.CaloEventSetup.HGCalV9Topology_cfi")
process.load("Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.HcalCommonData.hcalDDDRecConstants_cfi")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")
process.load("Geometry.HGCalGeometry.hgcalGeometryDump_cfi")

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D71.xml'),
                                            appendToDataLabel = cms.string('')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                appendToDataLabel = cms.string('')
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.Timing = cms.Service("Timing")

process.hgcalEEParametersInitialize.fromDD4Hep = True
process.hgcalHESiParametersInitialize.fromDD4Hep = True
process.hgcalHEScParametersInitialize.fromDD4Hep = True

process.p1 = cms.Path(process.hgcalGeometryDump)
