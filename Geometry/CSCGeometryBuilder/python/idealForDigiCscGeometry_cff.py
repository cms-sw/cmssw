import FWCore.ParameterSet.Config as cms

import Geometry.CSCGeometryBuilder.cscGeometry_cfi
#
# This cff provides a CSCGeometry with the label 'idealForDigi' that is for sure matching
# the ideal one and thus should be used in the digitisers.
#
idealForDigiCSCGeometry = Geometry.CSCGeometryBuilder.cscGeometry_cfi.CSCGeometryESModule.clone()
# The es_module providing fake (i.e. empty) alignment constants:
from Alignment.CommonAlignmentProducer.fakeForIdealAlignmentProducer_cfi import *
# replace idealForDigiCSCGeometry.applyAlignment = true # GF: See below
# Replace although false is default as protection against foreseen removal:
idealForDigiCSCGeometry.applyAlignment = False
# Label of the produced CSCGeometry:
idealForDigiCSCGeometry.appendToDataLabel = 'idealForDigi'
# Alignments are looked for with this label:
idealForDigiCSCGeometry.alignmentsLabel = 'fakeForIdeal'

