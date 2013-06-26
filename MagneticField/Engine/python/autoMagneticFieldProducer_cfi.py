import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use a field engine that is built using
# the current value provided in the ES. 

magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)

# avoid interference with EmptyESSource in uniformMagneticField.cfi
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


# Configue all possible slave (parametrized) fields

SlaveField0 = cms.ESProducer("UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double(0.0),
    label = cms.untracked.string('slave_0')
)

SlaveField20 = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('2_0T')
    ),
    label = cms.untracked.string('slave_20')
)

SlaveField30 = SlaveField20.clone()
SlaveField30.parameters.BValue = '3_0T'
SlaveField30.label = 'slave_30'

SlaveField35 = SlaveField20.clone()
SlaveField35.parameters.BValue = '3_5T'
SlaveField35.label = 'slave_35'

SlaveField38 = SlaveField20.clone()
SlaveField38.parameters.BValue= '3_8T'
SlaveField38.label = 'slave_38'

SlaveField40 = SlaveField20.clone()
SlaveField40.parameters.BValue= '4_0T'
SlaveField40.label = 'slave_40'


VBF0 = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    label = cms.untracked.string('0t'),
    useParametrizedTrackerField = cms.bool(True),
    paramLabel = cms.string('slave_0'),
    version = cms.string('grid_1103l_071212_2t'),
    geometryVersion = cms.int32(71212),
    debugBuilder = cms.untracked.bool(False),
    cacheLastVolume = cms.untracked.bool(True),
    scalingVolumes = cms.vint32(),
    scalingFactors = cms.vdouble(),
    gridFiles = cms.VPSet(
        cms.PSet( # Default tables, replicate sector 1
            volumes   = cms.string('1-312'),
            sectors   = cms.string('0') ,
            master    = cms.int32(1),
            path      = cms.string('grid.[v].bin'),
        ),
    )
)

VBF20 = VBF0.clone()
VBF20.version = 'grid_1103l_071212_2t'
VBF20.paramLabel = 'slave_20'
VBF20.label = '071212_2t'

VBF30 = VBF0.clone()
VBF30.version = 'grid_1103l_071212_3t'
VBF30.paramLabel = 'slave_30'
VBF30.label = '071212_3t'

VBF35 = VBF0.clone()
VBF35.version = 'grid_1103l_071212_3_5t'
VBF35.paramLabel = 'slave_35'
VBF35.label = '071212_3_5t'

#3.8T map: apply scaling factors; use sector-specific maps
from MagneticField.Engine.ScalingFactors_090322_2pi_090520_cfi import *
VBF38 = VBF0.clone()
VBF38.version = 'grid_1103l_090322_3_8t'
VBF38.geometryVersion = 90322
VBF38.paramLabel = 'slave_38'
VBF38.label = '090322_3_8t'
VBF38.scalingVolumes = fieldScaling.scalingVolumes
VBF38.scalingFactors = fieldScaling.scalingFactors
VBF38.gridFiles = cms.VPSet(
        cms.PSet( # Default tables, replicate sector 1
            volumes   = cms.string('1-312'),
            sectors   = cms.string('0') ,
            master    = cms.int32(1),
            path      = cms.string('grid.[v].bin'),
        ),

        cms.PSet( # Specific volumes in Barrel, sector 3
            volumes   = cms.string('176-186,231-241,286-296'),
            sectors   = cms.string('3') ,
            master    = cms.int32(3),
            path      = cms.string('S3/grid.[v].bin'),
        ),

        cms.PSet( # Specific volumes in Barrel, sector 4
            volumes   = cms.string('176-186,231-241,286-296'),
            sectors   = cms.string('4') ,
            master    = cms.int32(4),
            path      = cms.string('S4/grid.[v].bin'),
        ),

        cms.PSet(  # Specific volumes in Barrel and endcaps, sector 9
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('9') ,
            master    = cms.int32(9),
            path      = cms.string('S9/grid.[v].bin'),
        ),

        cms.PSet(  # Specific volumes in Barrel and endcaps, sector 10
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('10') ,
            master    = cms.int32(10),
            path      = cms.string('S10/grid.[v].bin'),
        ),
                                                        
        cms.PSet( # Specific volumes in Barrel and endcaps, sector 11
            volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
            sectors   = cms.string('11') ,
            master    = cms.int32(11),
            path      = cms.string('S11/grid.[v].bin'),
        ),
    )



VBF40 = VBF0.clone()
VBF40.version = 'grid_1103l_071212_4t'
VBF40.paramLabel = 'slave_40'
VBF40.label = '071212_4t'


AutoMagneticFieldESProducer = cms.ESProducer("AutoMagneticFieldESProducer",
   # if positive, set B value (in kGauss), overriding the current reading from DB
   valueOverride = cms.int32(-1),
   nominalCurrents = cms.untracked.vint32(-1, 0,9558,14416,16819,18268,19262),
   mapLabels = cms.untracked.vstring("090322_3_8t",
                                     "0t",
                                     "071212_2t",
                                     "071212_3t",
                                     "071212_3_5t",
                                     "090322_3_8t",
                                     "071212_4t"),
   label = cms.untracked.string(''),
 )

