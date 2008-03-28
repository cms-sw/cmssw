import FWCore.ParameterSet.Config as cms

# Producer for empty alignments
#==============================
from Alignment.CommonAlignmentProducer.fakeAlignmentProducer_cfi import *
# Source to put GlobalPositionRcd with unlimited IOV into ESetup
#=================================================================
fakeGlobalPositionSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('GlobalPositionRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


