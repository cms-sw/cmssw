import FWCore.ParameterSet.Config as cms

displacedTracks = cms.EDProducer("DuplicateListMerger",
    candidateComponents = cms.InputTag("duplicateDisplacedTrackCandidates","candidateMap"),
    candidateSource = cms.InputTag("duplicateDisplacedTrackCandidates","candidates"),
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    diffHitsCut = cms.int32(5),
    mergedMVAVals = cms.InputTag("duplicateDisplacedTrackClassifier","MVAValues"),
    mergedSource = cms.InputTag("mergedDuplicateDisplacedTracks"),
    mightGet = cms.optional.untracked.vstring,
    originalMVAVals = cms.InputTag("preDuplicateMergingDisplacedTracks","MVAValues"),
    originalSource = cms.InputTag("preDuplicateMergingDisplacedTracks"),
    trackAlgoPriorityOrder = cms.string('trackAlgoPriorityOrder')
)
