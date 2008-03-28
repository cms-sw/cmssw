# The following comments couldn't be translated into the new config version:

# avoid interference with EmptyESSource in uniformMagneticField.cfi

import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine.
from Geometry.CMSCommonData.cmsMagneticFieldXML_cfi import *
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")
VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    timerOn = cms.untracked.bool(False),
    useParametrizedTrackerField = cms.bool(False),
    findVolumeTolerance = cms.double(0.0),
    label = cms.untracked.string(''),
    version = cms.string('grid_85l_030919'),
    debugBuilder = cms.untracked.bool(False),
    cacheLastVolume = cms.untracked.bool(True)
)


