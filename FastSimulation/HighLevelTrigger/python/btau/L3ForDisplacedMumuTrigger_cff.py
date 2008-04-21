import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.GlobalPixelTracking_cff import *
hltCtfWithMaterialTracksMumu = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("hltL3Muons")),
    ptMin = cms.untracked.double(3.0),
    minHits = cms.untracked.uint32(5)
)

hltMuTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumu"),
    particleType = cms.string('mu-')
)

Mumutracks = cms.Sequence(cms.SequencePlaceholder("l3muonreco")+hltCtfWithMaterialTracksMumu)
Mumucand = cms.Sequence(hltMuTracks)
l3displacedMumureco = cms.Sequence(cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("recopixelvertexing")+Mumutracks+Mumucand)

