import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.GlobalPixelTracking_cff import *
hltCtfWithMaterialTracksMumuk = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(3.0),
    minHits = cms.untracked.uint32(5)
)

hltMumukAllConeTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumuk"),
    particleType = cms.string('mu-')
)

hltMumukTracking = cms.Sequence(cms.SequencePlaceholder("l3muonreco")+hltCtfWithMaterialTracksMumuk)
hltMumukCandidates = cms.Sequence(hltMumukAllConeTracks)
l3MumukReco = cms.Sequence(cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("recopixelvertexing")+hltMumukTracking+hltMumukCandidates)

