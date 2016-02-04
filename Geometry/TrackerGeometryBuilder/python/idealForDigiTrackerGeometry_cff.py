import FWCore.ParameterSet.Config as cms

import Geometry.TrackerGeometryBuilder.trackerGeometry_cfi
#
# This cff provides a TrackerGeometry with the label 'idealForDigi' that is for sure matching
# the ideal one and thus should be used in the digitisers.
#
idealForDigiTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerGeometry_cfi.TrackerDigiGeometryESModule.clone()
# The es_module providing fake (i.e. empty) alignment constants:
from Alignment.CommonAlignmentProducer.fakeForIdealAlignmentProducer_cfi import *
# need to set to False, see below:
idealForDigiTrackerGeometry.applyAlignment = False
# Label of the produced TrackerGeometry:
idealForDigiTrackerGeometry.appendToDataLabel = 'idealForDigi'
# Alignments are looked for with this label:
idealForDigiTrackerGeometry.alignmentsLabel = 'fakeForIdeal'

#fakeTrackerAlignmentSource = cms.ESSource("EmptyESSource",
#                                          recordName = cms.string("TrackerAlignmentRcd"),
#                                          firstValid = cms.vuint32(1),
#                                          iovIsRunNotTime = cms.bool(True)
#                                          )
#The follwoing to be converted as the above...
#es_source fakeTrackerAlignmentErrorSource = EmptyESSource {
#    string recordName = "TrackerAlignmentErrorRcd"
#    vuint32 firstValid = {1}
#    bool iovIsRunNotTime = true
#}
## care: This might lead to a duplication with DT and CSC equivalents of this file:
#es_source fakeGlobalPositionSource = EmptyESSource {
#    string recordName = "GlobalPositionRcd"
#    vuint32 firstValid = {1}
#    bool iovIsRunNotTime = true
#}

# Comments by GF:
# - In anticipation of the removal of the applyAlignment flag, I'd like to keep it true.
# - Then we would need to get IOVs for the fake alignments,
#   * either using FakeAlignmentSource instead of FakeAlignmentProducer in fakeForIdealAlignmentProducer.cfi
#   * or by using the commented IOV settings above.
# - But this causes problems in co-existence with e.g. GlobalTag: 
#   Both Globaltag as well as FakeAlignmentSource provide IOV - it is not distinguished to provide IOV for 
#   a given label only (e.g. 'fakeForIdeal' compared to '').
# - I'll try to contact framework people for CMSSW_2_2_0 and above.
