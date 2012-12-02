import FWCore.ParameterSet.Config as cms
import os

duplicateTrackMerger = cms.EDProducer("DuplicateTrackMerger",
                                      source = cms.InputTag("preDuplicateMergingGeneralTracks"),
                                      minDeltaR3d = cms.double(-4.0),
                                      minBDTG = cms.double(-0.1),
                                      minpT = cms.double(0.2),
                                      minP = cms.double(0.4),
                                      maxDCA = cms.double(50.0),
                                      maxDPhi = cms.double(0.35),
                                      maxDLambda = cms.double(0.35),
                                      maxDdsz = cms.double(20.0),
                                      maxDdxy = cms.double(20.0),
                                      maxDQoP = cms.double(0.25),
                                      #weightsFile=cms.string(os.getenv("CMSSW_BASE")+"/src/RecoTracker/FinalTrackSelectors/data/DuplicateWeights.xml"),
                                      useInnermostState  = cms.bool(True),
                                      ttrhBuilderName    = cms.string("WithAngleAndTemplate")
                                      )

duplicateListMerger = cms.EDProducer("DuplicateListMerger",
                                     originalSource = cms.InputTag("preDuplicateMergingGeneralTracks"),
                                     diffHitsCut = cms.int32(5),
                                     minTrkProbCut = cms.double(0.0),
                                     mergedSource = cms.InputTag("mergedDuplicateTracks"),
                                     candidateSource = cms.InputTag("duplicateTrackMerger","candidateMap"),
                                     newQuality = cms.string('confirmed')
                                     )
