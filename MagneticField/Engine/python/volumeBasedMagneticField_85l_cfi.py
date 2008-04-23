import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine version 85l

from MagneticField.GeomBuilder.cmsMagneticFieldXML_85l_cfi import *

# avoid interference with EmptyESSource in uniformMagneticField.cfi
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_85l_030919'),
    parameters = cms.PSet(
        a = cms.double(4.643),
        b0 = cms.double(40.681),
        l = cms.double(15.284)
    ),
    label = cms.untracked.string('parametrizedField')
)


VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    timerOn = cms.untracked.bool(False),
    useParametrizedTrackerField = cms.bool(False),
    label = cms.untracked.string(''),
    version = cms.string('grid_85l_030919'),
    debugBuilder = cms.untracked.bool(False),
    cacheLastVolume = cms.untracked.bool(True)
)


