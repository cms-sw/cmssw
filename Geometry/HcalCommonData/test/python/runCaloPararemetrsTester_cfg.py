import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Geometry.HcalCommonData.hcalParameters_cfi')
process.load('Geometry.HcalCommonData.hcalSimulationParameters_cfi')
process.load('Geometry.HcalCommonData.caloSimulationParameters_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.Geometry=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/HcalAlgo/data/cms-test-ddhcalHF-algorithm.xml'),
                                            appendToDataLabel = cms.string('')
                                            )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                appendToDataLabel = cms.string('')
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.caloSimulationParameters.fromDD4Hep = cms.bool(True)

process.hpa = cms.EDAnalyzer("CaloSimParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
