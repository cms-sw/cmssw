import FWCore.ParameterSet.Config as cms

process = cms.Process("DDG4ProdCutsTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'g4prodcuts'),
    categories = cms.untracked.vstring('Geometry'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    g4prodcuts = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        Geometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('cout',
                                         'g4prodcuts')
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            appendToDataLabel = cms.string('MUON')
                                        )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('MUON')
                                                 )

process.test = cms.EDAnalyzer("DDTestSpecParsFilter",
                              DDDetector = cms.ESInputTag('','MUON'),
                              attribute = cms.untracked.string('CMSCutsRegion'),
                              value = cms.untracked.string('Muon')
                          )

process.p = cms.Path(process.test)
