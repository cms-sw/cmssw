
import FWCore.ParameterSet.Config as cms

FakeAlignmentSource = cms.ESSource("FakeAlignmentSource",
    produceTracker = cms.bool(True),
    produceDT = cms.bool(True),
    produceCSC = cms.bool(True),
    produceGlobalPosition = cms.bool(True),
    produceTrackerSurfaceDeformation = cms.bool(True),
    appendToDataLabel = cms.string('')
)
