import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine version 071212 at 2T.
#
# WARNING: THIS FIELD MAP IS OBSOLETE, EXCEPT FOR THE WORKING POINT AT 2T.
# If in doubt, use the standard sequence Configuration.StandardSequences.MagneticField_cff 

magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)

# avoid interference with EmptyESSource in uniformMagneticField.cfi
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('2_0T')
    ),
    label = cms.untracked.string('parametrizedField')
)


VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    useParametrizedTrackerField = cms.bool(True),
    label = cms.untracked.string(''),
    paramLabel = cms.string('parametrizedField'),
    version = cms.string('grid_1103l_071212_2t'),
    geometryVersion = cms.int32(71212),
    debugBuilder = cms.untracked.bool(False),
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


