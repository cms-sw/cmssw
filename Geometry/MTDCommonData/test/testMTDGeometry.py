import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("MTDGeometryTest",Phase2C11I13M9,dd4hep)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    ),
    files = cms.untracked.PSet(
        mtdGeometry = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            FWKINFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            Geometry = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            enableStatistics = cms.untracked.bool(True),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MTDCommonData/data/dd4hep/cms-mtdD76-geometry.xml'),
                                            appendToDataLabel = cms.string('')
                                            )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('')
                                                     )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              DDDetector = cms.ESInputTag('','')
                              )

process.dump = cms.EDAnalyzer("DDTestDumpFile",
                              DDDetector = cms.ESInputTag('','')
                              )

process.p = cms.Path(process.test+process.dump)
