import FWCore.ParameterSet.Config as cms

import Geometry.MTDGeometryBuilder.mtdGeometry_cfi
#
# This cff provides a TrackerGeometry with the label 'idealForDigi' that is for sure matching
# the ideal one and thus should be used in the digitisers.
#
idealForDigiMTDGeometry = Geometry.MTDGeometryBuilder.mtdGeometry_cfi.mtdGeometry.clone()
# The es_module providing fake (i.e. empty) alignment constants:
from Alignment.CommonAlignmentProducer.fakeForIdealAlignmentProducer_cfi import *
# need to set to False, see below:
idealForDigiMTDGeometry.applyAlignment = False
# Label of the produced TrackerGeometry:
idealForDigiMTDGeometry.appendToDataLabel = 'idealForDigi'
# Alignments are looked for with this label:
idealForDigiMTDGeometry.alignmentsLabel = 'fakeForIdeal'
