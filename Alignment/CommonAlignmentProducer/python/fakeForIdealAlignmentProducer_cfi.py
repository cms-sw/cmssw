import FWCore.ParameterSet.Config as cms

import Alignment.CommonAlignmentProducer.fakeAlignmentProducer_cfi
# Just 'append' a data label to differentiate from 'normal' fake alignment:
fakeForIdealAlignment = Alignment.CommonAlignmentProducer.fakeAlignmentProducer_cfi.fakeAlignment.clone()
fakeForIdealAlignment.appendToDataLabel = 'fakeForIdeal'

