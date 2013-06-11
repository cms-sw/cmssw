import FWCore.ParameterSet.Config as cms

import Geometry.TrackerGeometryBuilder.trackerSLHCGeometryDB_cfi
#
# This cff provides a TrackerGeometry with the label 'idealForDigi' that is for sure matching
# the ideal one and thus should be used in the digitisers.
#
idealForDigiTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerSLHCGeometryDB_cfi.trackerSLHCGeometryDB.clone()
# The es_module providing fake (i.e. empty) alignment constants:
from Alignment.CommonAlignmentProducer.fakeForIdealAlignmentProducer_cfi import *
# need to set to False, see below:
idealForDigiTrackerGeometry.applyAlignment = False
# Label of the produced TrackerGeometry:
idealForDigiTrackerGeometry.appendToDataLabel = 'idealForDigi'
# Alignments are looked for with this label:
idealForDigiTrackerGeometry.alignmentsLabel = 'fakeForIdeal'

import Geometry.TrackerGeometryBuilder.trackerSLHCGeometryConstants_cfi as trackerGeometryConstants_cfi
idealForDigiTrackerGeometry.trackerGeometryConstants = cms.PSet(trackerGeometryConstants_cfi.trackerGeometryConstants)

