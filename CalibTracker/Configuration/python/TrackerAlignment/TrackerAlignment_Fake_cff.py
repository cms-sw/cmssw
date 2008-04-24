import FWCore.ParameterSet.Config as cms

# Producer module
# ===============
# Dummy alignment producer (creating empty Aligments and AlignmentErrors)
from Alignment.CommonAlignmentProducer.fakeAlignmentProducer_cfi import *
# Sources that actually put the records into ESetup
# (Here with unlimited IOV, taken from producer above.)
# =====================================================
from Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff import *
fakeTrackerAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('TrackerAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeTrackerAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('TrackerAlignmentErrorRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


