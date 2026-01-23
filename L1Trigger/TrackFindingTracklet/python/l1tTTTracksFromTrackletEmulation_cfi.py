import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params
from L1Trigger.TrackFindingTracklet.ChannelAssignment_cff import ChannelAssignment
from L1Trigger.TrackerTFP.LayerEncoding_cff import TrackTriggerLayerEncoding

l1tTTTracksFromTrackletEmulation = cms.EDProducer("L1FPGATrackProducer",
                                               TrackFindingTrackletProducer_params,
                                               TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                               InputTagTTDTC = cms.InputTag("ProducerDTC", "StubAccepted"),
                                               readMoreMcTruth = cms.bool(False),
                                               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                               asciiFileName = cms.untracked.string(""),
                                               FailScenario = cms.untracked.int32(0),
                                               Extended = cms.bool(False),
                                               Reduced = cms.bool(False),
                                               Hnpar = cms.uint32(4),
                                               # These 3 files only used for reduced mode and set in Customize_cff.py
                                               memoryModulesFile = cms.string(""),
                                               processingModulesFile = cms.string(""),
                                               wiresFile = cms.string(""),
                                               wiresJSONFile = cms.string("L1Trigger/TrackFindingTracklet/data/seedWiring.json"),
                                               # Quality Flag and Quality params
                                               TrackQuality = cms.bool(True),
                                               Fakefit = cms.bool(False), # True causes Tracklet reco to output TTTracks before DR & KF
                                               StoreTrackBuilderOutput = cms.bool(False), # if True EDProducts for TrackBuilder tracks and stubs will be filled
                                               RemovalType = cms.string("merge"), # Duplicate track removal
                                               DoMultipleMatches = cms.bool(True), # Allow tracklet tracks multiple stubs per layer
                                               # TQ
                                               # It is compatible with the HYBRID simulation and will give equivilant performance with this workflow
                                               Model = cms.FileInPath( "L1Trigger/TrackTrigger/data/L1_TrackQuality_GBDT_emulation_digitized.json" ),
                                               #Vector of strings of training features, in the order that the model was trained with
                                               FeatureNames = cms.vstring( ["tanl",
                                                                           "z0_scaled",
                                                                           "bendchi2_bin",
                                                                           "nstub",
                                                                           "nlaymiss_interior",
                                                                           "chi2rphi_bin",
                                                                           "chi2rz_bin"
                                                                           ] )
    )

l1tTTTracksFromExtendedTrackletEmulation = l1tTTTracksFromTrackletEmulation.clone(
                                               Extended = cms.bool(True),
                                               Reduced = cms.bool(False),
                                               Hnpar = cms.uint32(5),
                                               # Quality Flag and Quality params
                                               TrackQuality = cms.bool(False)
    )
