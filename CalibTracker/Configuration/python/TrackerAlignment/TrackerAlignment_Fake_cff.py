import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.fakeAlignmentProducer_cfi import *
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


