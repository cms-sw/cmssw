import FWCore.ParameterSet.Config as cms

TTTracksFromTrackletEmulation = cms.EDProducer("L1FPGATrackProducer",
                                               TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                               readMoreMcTruth = cms.bool(True),
                                               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                               asciiFileName = cms.untracked.string(""),
                                               # (if running on CRAB use "../../fitpattern.txt" etc instead)
                                               Extended=cms.bool(False),
                                               Hnpar=cms.uint32(4),
                                               fitPatternFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/fitpattern.txt'),
                                               memoryModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/memorymodules_hourglass.dat'),
                                               processingModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/processingmodules_hourglass.dat'),
                                               wiresFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/wires_hourglass.dat'),
                                               DTCLinkFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/calcNumDTCLinks.txt'),
                                               DTCLinkLayerDiskFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/dtclinklayerdisk.dat'),
                                               moduleCablingFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/modules_T5v3_27SP_nonant_tracklet.dat')
    )

TTTracksFromExtendedTrackletEmulation = TTTracksFromTrackletEmulation.clone(
                                               Extended=cms.bool(True),
                                               Hnpar=cms.uint32(5),
                                               memoryModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/memorymodules_hourglassExtended.dat'),
                                               processingModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/processingmodules_hourglassExtended.dat'),
                                               wiresFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/wires_hourglassExtended.dat'),
    )

