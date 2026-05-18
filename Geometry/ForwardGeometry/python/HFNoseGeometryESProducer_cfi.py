import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HFNose Geometry
#

from Geometry.HGCalGeometry.hgcalEEGeometryESProducer_cfi import *


hfNoseGeometryESProducer = hgcalEEGeometryESProducer.clone(
    name = "HGCalHFNoseSensitive"
)
