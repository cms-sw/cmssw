import FWCore.ParameterSet.Config as cms

preDuplicateMergingDisplacedTracks = cms.EDProducer("TrackCollectionMerger",
    allowFirstHitShare = cms.bool(True),
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    enableMerging = cms.bool(True),
    foundHitBonus = cms.double(100.0),
    inputClassifiers = cms.vstring('muonSeededTracksOutInDisplacedClassifier'),
    lostHitPenalty = cms.double(1.0),
    mightGet = cms.optional.untracked.vstring,
    minQuality = cms.string('loose'),
    minShareHits = cms.uint32(2),
    shareFrac = cms.double(0.19),
    trackAlgoPriorityOrder = cms.string('trackAlgoPriorityOrder'),
    trackProducers = cms.VInputTag("muonSeededTracksOutInDisplaced")
)
