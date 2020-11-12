import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSimParametersTest")

process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('EcalGeom')
    process.MessageLogger.categories.append('EcalSim')
    process.MessageLogger.categories.append('Geometry')

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

process.ecalSimulationParametersEB.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersEE.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersES.fromDD4Hep = cms.bool(True)

process.load('Geometry.EcalCommonData.ecalSimulationParametersAnalyzer_cff')

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.ecalSimulationParametersAnalyzerEB+process.ecalSimulationParametersAnalyzerEE+process.ecalSimulationParametersAnalyzerES)
