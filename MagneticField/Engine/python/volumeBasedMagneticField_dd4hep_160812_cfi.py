import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticField_160812_cfi import VBFConfig_160812 
from MagneticField.Engine.volumeBasedMagneticField_160812_cfi import ParametrizedMagneticFieldProducer


# This cfi contains everything needed to use the VolumeBased magnetic
# field engine version 160812 built using dd4hep for the geometry.
#
# PLEASE DO NOT USE THIS DIRECTLY
# Always use the standard sequence Configuration.StandardSequences.MagneticField_cff


DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath('MagneticField/GeomBuilder/data/cms-mf-geometry_160812.xml'),
    rootDDName = cms.string('cmsMagneticField:MAGF'),
    appendToDataLabel = cms.string('magfield')
    )


DDCompactViewMFESProducer = cms.ESProducer("DDCompactViewMFESProducer",
                                            appendToDataLabel = cms.string('magfield')
                                           )


VolumeBasedMagneticFieldESProducer = cms.ESProducer("DD4hep_VolumeBasedMagneticFieldESProducer",
    VBFConfig_160812,
    DDDetector = cms.ESInputTag('', 'magfield'),
    appendToDataLabel = cms.string(''),
)


### To set a different nominal map, set the following in your .py:

### 3T
#VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3t')
#ParametrizedMagneticFieldProducer.parameters.BValue = cms.string('3_0T')

### 3.5T
#VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3_5t')
#ParametrizedMagneticFieldProducer.parameters.BValue = cms.string('3_5T')


### Run I, 3.8T
#VolumeBasedMagneticFieldESProducer.version = cms.string('grid_160812_3_8t_Run1')
