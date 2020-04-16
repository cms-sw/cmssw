import FWCore.ParameterSet.Config as cms

process = cms.Process("DDSpecParsTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'specparfiltered'),
    categories = cms.untracked.vstring('Geometry'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    specparfiltered = cms.untracked.PSet(
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
                                         'specparfiltered')
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-geometry-2021.xml'),
                                            appendToDataLabel = cms.string('CMS')
                                            )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('CMS')
                                                     )

process.test = cms.EDAnalyzer("DDTestSpecParsFilter",
                              DDDetector = cms.ESInputTag('','CMS'),
                              attribute = cms.untracked.string('TrackingMaterialGroup'), ###MuStructure'),
                              value = cms.untracked.string('') ### MuonEndcapCSC') ###'MuonBarrelDT')
                              )

process.p = cms.Path(process.test)
