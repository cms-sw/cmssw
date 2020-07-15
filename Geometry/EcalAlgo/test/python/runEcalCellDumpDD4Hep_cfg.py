import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalGeometryTest")

process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.HcalCommonData.hcalDDDRecConstants_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi");
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/EcalCommonData/data/dd4hep/cms-ecal-geometry.xml'),
                                           appendToDataLabel = cms.string('')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                appendToDataLabel = cms.string('')
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hcalParameters.fromDD4Hep = cms.bool(True)
process.caloSimulationParameters.fromDD4Hep = cms.bool(True)
process.CaloGeometryBuilder.SelectedCalos = ['EcalBarrel', 'EcalEndcap', 'EcalPreshower']
process.ecalSimulationParametersEB.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersEE.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersES.fromDD4Hep = cms.bool(True)

process.demo1 = cms.EDAnalyzer("EcalBarrelCellParameterDump")
process.demo2 = cms.EDAnalyzer("EcalEndcapCellParameterDump")
process.demo3 = cms.EDAnalyzer("EcalPreshowerCellParameterDump")

process.p1 = cms.Path(process.demo1 * process.demo2 * process.demo3)
