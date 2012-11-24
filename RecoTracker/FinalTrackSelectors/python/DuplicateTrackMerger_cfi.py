import FWCore.ParameterSet.Config as cms
import os

duplicateTrackMerger = cms.EDProducer("DuplicateTrackMerger",
                                      source = cms.InputTag("preDuplicateMergingGeneralTracks"),
                                      minDeltaR3d = cms.double(-4.0),
                                      minBDTG = cms.double(-0.96),
                                      #weightsFile=cms.string(os.getenv("CMSSW_BASE")+"/src/RecoTracker/FinalTrackSelectors/data/DuplicateWeights.xml"),
                                      useInnermostState  = cms.bool(True),
                                      ttrhBuilderName    = cms.string("WithAngleAndTemplate")
                                      )

duplicateListMerger = cms.EDProducer("DuplicateListMerger",
                                     originalSource = cms.InputTag("preDuplicateMergingGeneralTracks"),
                                     diffHitsCut = cms.int32(5),
                                     mergedSource = cms.InputTag("mergedDuplicateTracks"),
                                     candidateSource = cms.InputTag("duplicateTrackMerger","candidateMap"),
                                     newQuality = cms.string('confirmed')
                                     )
