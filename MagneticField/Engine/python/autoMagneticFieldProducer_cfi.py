import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use a field engine that is built using
# the current value provided in the ES. 

from MagneticField.GeomBuilder.cmsMagneticFieldXML_1103l_cfi import *

# avoid interference with EmptyESSource in uniformMagneticField.cfi
es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


AutoMagneticFieldESProducer = cms.ESProducer("AutoMagneticFieldESProducer",
   # TEMPORARY
   value = cms.untracked.int32(38),

   model = cms.string('grid_1103l_071212'),
   useParametrizedTrackerField = cms.bool(True),
   subModel = cms.string('OAE_1103l_071212'),
   label = cms.untracked.string('')
)

