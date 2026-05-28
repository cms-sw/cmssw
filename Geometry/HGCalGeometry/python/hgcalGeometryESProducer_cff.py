import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Geometry
#

from Geometry.HGCalGeometry.hgcalEEGeometryESProducer_cfi import *


hgcalHESilGeometryESProducer = hgcalEEGeometryESProducer.clone(
    name = "HGCalHESiliconSensitive"
)

hgcalHESciGeometryESProducer = hgcalEEGeometryESProducer.clone(
    name = "HGCalHEScintillatorSensitive"
)

