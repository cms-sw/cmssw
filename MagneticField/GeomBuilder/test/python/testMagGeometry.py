import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
    )

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring( '*' ),
    destinations = cms.untracked.vstring('cout'),
    categories     = cms.untracked.vstring ( '*' ),
    cout = cms.untracked.PSet(
      noLineBreaks = cms.untracked.bool(True),
      INFO  =  cms.untracked.PSet (limit = cms.untracked.int32(-1)),
      DEBUG =  cms.untracked.PSet (limit = cms.untracked.int32(-1)),
      WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
      ),
      ERROR = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
      ),
      threshold = cms.untracked.string('DEBUG'),
      default =  cms.untracked.PSet (limit = cms.untracked.int32(-1))
    )
)


process.ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('3_5T')
    ),
    label = cms.untracked.string('parametrizedField')
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-mf-geometry.xml'),
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
                                              paramLabel = cms.string('parametrizedField'),
                                              version = cms.string('grid_160812_3_5t'),
                                              geometryVersion = cms.int32(160812),
                                              debugBuilder = cms.untracked.bool(True),
                                              cacheLastVolume = cms.untracked.bool(True),
                                              scalingVolumes = cms.vint32(),
                                              scalingFactors = cms.vdouble(),

                                              gridFiles = cms.VPSet(
                                                          cms.PSet(
                                                               volumes   = cms.string('1001-1010,1012-1027,1030-1033,1036-1041,1044-1049,1052-1057,1060-1063,1066-1071,1074-1077,1080-1097,1102-1129,1138-1402,1415-1416,' + 
                                                                                      '2001-2010,2012-2027,2030-2033,2036-2041,2044-2049,2052-2057,2060-2063,2066-2071,2074-2077,2080-2097,2102-2129,2138-2402,2415-2416'),
                                                               sectors   = cms.string('0') ,
                                                               master    = cms.int32(0),
                                                               path      = cms.string('s[s]/grid.[v].bin'),
                                                          ),
                                              # Replicate sector 1 for volumes outside any detector
                                                          cms.PSet(
                                                               volumes   = cms.string('1011,1028-1029,1034-1035,1042-1043,1050-1051,1058-1059,1064-1065,1072-1073,1078-1079,'+ # volumes extending from R~7.6 m to to R=9 m,
                                                                                      '1098-1101,1130-1137,' + # Forward volumes, ouside CASTOR/HF
                                                                                      '1403-1414,1417-1464,' # Volumes beyond |Z|>17.74
                                                                                      '2011,2028-2029,2034-2035,2042-2043,2050-2051,2058-2059,2064-2065,2072-2073,2078-2079,'+
                                                                                      '2098-2101,2130-2137,'+
                                                                                      '2403-2414,2417-2464'),
                                                               sectors   = cms.string('0'),
                                                               master    = cms.int32(1),
                                                               path      = cms.string('s01/grid.[v].bin'),
                                                          ),
                                                     )
                                               )

process.DDCompactViewMFESProducer = cms.ESProducer("DDCompactViewMFESProducer",
                                                 appendToDataLabel = cms.string('magfield')
                                                )

process.test = cms.EDAnalyzer("testMagGeometryAnalyzer",
                              DDDetector = cms.ESInputTag('', 'magfield')
                              )

process.p = cms.Path(process.test)
