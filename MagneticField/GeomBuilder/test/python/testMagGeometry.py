import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        enable = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)


process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('MagneticField/GeomBuilder/data/cms-mf-geometry_160812.xml'),
                                            rootDDName = cms.string('cmsMagneticField:MAGF'),
                                            appendToDataLabel = cms.string('magfield')
                                            )

process.MagneticFieldESProducer = cms.ESProducer("DD4hep_VolumeBasedMagneticFieldESProducer",
                                              DDDetector = cms.ESInputTag('', 'magfield'),
                                              appendToDataLabel = cms.string(''),
                                              useParametrizedTrackerField = cms.bool(False),
                                              label = cms.untracked.string(''),
                                              attribute = cms.string('magfield'),
                                              value = cms.string('magfield'),
                                              paramLabel = cms.string(''),
                                              version = cms.string('fake'),
                                              geometryVersion = cms.int32(160812),
                                              debugBuilder = cms.untracked.bool(False), # Set to True to activate full debug output
                                              scalingVolumes = cms.vint32(),
                                              scalingFactors = cms.vdouble(),

                                              gridFiles = cms.VPSet()
                                               )


# Uncomment this to test geometry version 71212
#process.DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath('MagneticField/GeomBuilder/data/cms-mf-geometry_71212.xml')
#process.MagneticFieldESProducer.geometryVersion = cms.int32(71212)



process.DDCompactViewMFESProducer = cms.ESProducer("DDCompactViewMFESProducer",
                                                 appendToDataLabel = cms.string('magfield')
                                                )

process.test = cms.EDAnalyzer("testMagGeometryAnalyzer",
                              DDDetector = cms.ESInputTag('', 'magfield')
                              )

process.p = cms.Path(process.test)
