import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use a field engine that is built using
# the current value provided in the ES. 

magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldParameters_07.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)

# avoid interference with EmptyESSource in uniformMagneticField.cfi
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


VolumeBasedMagneticFieldESProducer = cms.ESProducer("AutoMagneticFieldESProducer",
   # if positive, set B value (in kGauss), overriding the current reading from DB
   valueOverride = cms.int32(-1),

   model = cms.string('grid_1103l_071212'),
   useParametrizedTrackerField = cms.bool(True),
   subModel = cms.string('OAE_1103l_071212'),
   label = cms.untracked.string(''),
   scalingVolumes = cms.vint32(),
   scalingFactors = cms.vdouble()
   # Other VBF card are taken with their default value...
   # cacheLastVolume = cms.untracked.bool(True),
   # debugBuilder = cms.untracked.bool(False)
 )

