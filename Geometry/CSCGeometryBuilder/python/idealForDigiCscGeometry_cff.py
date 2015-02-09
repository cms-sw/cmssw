import FWCore.ParameterSet.Config as cms

import Geometry.CSCGeometryBuilder.cscGeometry_cfi
#
# This cff provides a CSCGeometry with the label 'idealForDigi' that is for sure matching
# the ideal one and thus should be used in the digitisers.
#
idealForDigiCSCGeometry = Geometry.CSCGeometryBuilder.cscGeometry_cfi.CSCGeometryESModule.clone()
# The es_module providing fake (i.e. empty) alignment constants:
from Alignment.CommonAlignmentProducer.fakeForIdealAlignmentProducer_cfi import *
# need to set to False, see below:
idealForDigiCSCGeometry.applyAlignment = False
# Label of the produced CSCGeometry:
idealForDigiCSCGeometry.appendToDataLabel = 'idealForDigi'
# Alignments are looked for with this label:
idealForDigiCSCGeometry.alignmentsLabel = 'fakeForIdeal'
# would need conversion
#es_source fakeCSCAlignmentSource = EmptyESSource {
#    string recordName = "CSCAlignmentRcd"
#    vuint32 firstValid = {1}
#    bool iovIsRunNotTime = true
#}
#es_source fakeCSCAlignmentErrorSource = EmptyESSource {
#    string recordName = "CSCAlignmentErrorExtendedRcd"
#    vuint32 firstValid = {1}
#    bool iovIsRunNotTime = true
#}
## care: This might lead to a duplication with DT and tracker equivalents of this file:
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
