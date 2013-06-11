import FWCore.ParameterSet.Config as cms


# This cfi contains everything needed to use the VolumeBased magnetic
# field engine version 120812

magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
#FIXME: material description is for the LARGE version!!
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1_v7_large.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_2_v7_large.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)

# avoid interference with EmptyESSource in uniformMagneticField.cfi
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('3_8T')
    ),
    label = cms.untracked.string('parametrizedField')
)


VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    useParametrizedTrackerField = cms.bool(True),
    label = cms.untracked.string(''),
    paramLabel = cms.string('parametrizedField'),
    version = cms.string('grid_120812_3_8t_v7_small'),
    geometryVersion = cms.int32(120812),
    debugBuilder = cms.untracked.bool(False),
    cacheLastVolume = cms.untracked.bool(True),
    scalingVolumes = cms.vint32(),
    scalingFactors = cms.vdouble(),


    gridFiles = cms.VPSet(
         # Default: specific tables for all volumes
#          cms.PSet(
#              volumes   = cms.string('1001-1360'),
#              sectors   = cms.string('0') ,
#              master    = cms.int32(0),
#              path      = cms.string('s[s]_1/grid.[v].bin'),
#          ),
#          cms.PSet(
#              volumes   = cms.string('2001-2360'),
#              sectors   = cms.string('0'),
#              master    = cms.int32(0),
#              path      = cms.string('s[s]_2/grid.[v].bin'),
#          ),

          # Specific tables for each sector
          cms.PSet(
              volumes   = cms.string('1001-1010,1012-1027,1030-1033,1036-1041,1044-1049,1052-1057,1060-1063,1066-1071,1074-1077,1080-1083,1130-1133,1138-1360'),
              sectors   = cms.string('0') ,
              master    = cms.int32(0),
              path      = cms.string('s[s]_1/grid.[v].bin'),
          ),
          cms.PSet(
              volumes   = cms.string('2001-2010,2012-2027,2030-2033,2036-2041,2044-2049,2052-2057,2060-2063,2066-2071,2074-2077,2080-2083,2130-2133,2138-2360'),
              sectors   = cms.string('0'),
              master    = cms.int32(0),
              path      = cms.string('s[s]_2/grid.[v].bin'),
          ),

         # Replicate sector 1 for volumes outside any detector
#           cms.PSet(
#              volumes   = cms.string('1011,1028-1029,1034-1035,1042-1043,1050-1051,1058-1059,1064-1065,1072-1073,1078-1079,1084-1129,1134-1137'),
#              sectors   = cms.string('0'),
#              master    = cms.int32(1),
#              path      = cms.string('s01_1/grid.[v].bin'),
#          ),
#          cms.PSet(
#              volumes   = cms.string('2011,2028-2029,2034-2035,2042-2043,2050-2051,2058-2059,2064-2065,2072-2073,2078-2079,2084-2129,2134-2137'),
#              sectors   = cms.string('0'),
#              master    = cms.int32(1),
#              path      = cms.string('s01_2/grid.[v].bin'),
#          ),   

           cms.PSet(
              volumes   = cms.string('1011,1028-1029,1034-1035,1042-1043,1050-1051,1058-1059,1064-1065,1072-1073,1078-1079,1084-1129,1136-1137'),
              sectors   = cms.string('0'),
              master    = cms.int32(1),
              path      = cms.string('s01_1/grid.[v].bin'),
          ),
          cms.PSet(
              volumes   = cms.string('2011,2028-2029,2034-2035,2042-2043,2050-2051,2058-2059,2064-2065,2072-2073,2078-2079,2084-2129,2136-2137'),
              sectors   = cms.string('0'),
              master    = cms.int32(1),
              path      = cms.string('s01_2/grid.[v].bin'),
          ),   

         # Replicate sector 4 for the volume outside CASTOR, to avoid aliasing due to the plates in the cylinder gap
         # between the collar and the rotating shielding.
         cms.PSet(
             volumes   = cms.string('1134-1135'),
             sectors   = cms.string('0'),
             master    = cms.int32(4),
             path      = cms.string('s04_1/grid.[v].bin'),
         ),
         cms.PSet(
             volumes   = cms.string('2134-2135'),
             sectors   = cms.string('0'),
             master    = cms.int32(4),
             path      = cms.string('s04_2/grid.[v].bin'),
         ),


     )
)


