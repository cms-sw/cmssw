import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C4_cff import Phase2C4

process = cms.Process("HcalParametersTest",Phase2C4)

process.load('Geometry.HcalCommonData.hcalParameters_cff')
process.load('Geometry.HcalCommonData.hcalSimulationParameters_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D43.xml'),
                                            appendToDataLabel = cms.string('')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                appendToDataLabel = cms.string('')
)

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")
process.hcalParameters.fromDD4Hep = cms.bool(True)
process.hcalSimulationParameters.fromDD4Hep = cms.bool(True)

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
