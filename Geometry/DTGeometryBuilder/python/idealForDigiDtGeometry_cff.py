import FWCore.ParameterSet.Config as cms

import Geometry.DTGeometryBuilder.dtGeometry_cfi
#
# This cff provides a DTGeometry with the label 'idealForDigi' that is for sure matching
# the ideal one and thus should be used in the digitisers.
#
idealForDigiDTGeometry = Geometry.DTGeometryBuilder.dtGeometry_cfi.DTGeometryESModule.clone()
# The es_module providing fake (i.e. empty) alignment constants:
from Alignment.CommonAlignmentProducer.fakeForIdealAlignmentProducer_cfi import *
#replace idealForDigiDTGeometry.applyAlignment = true # GF: See below
# Replace although false is default as protection against foreseen removal:
idealForDigiDTGeometry.applyAlignment = False
# Label of the produced DTGeometry:
idealForDigiDTGeometry.appendToDataLabel = 'idealForDigi'
# Alignments are looked for with this label:
idealForDigiDTGeometry.alignmentsLabel = 'fakeForIdeal'

