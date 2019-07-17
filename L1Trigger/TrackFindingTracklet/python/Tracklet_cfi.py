import FWCore.ParameterSet.Config as cms

TTTracksFromTracklet = cms.EDProducer("L1TrackProducer",
                                      SimTrackSource = cms.InputTag("g4SimHits"),
                                      SimVertexSource = cms.InputTag("g4SimHits"),
                                      TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                      MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                      MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                      TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                      TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                      BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                      asciiFileName = cms.untracked.string(""),
                                      failscenario = cms.untracked.int32(0),
                                      trackerGeometryType  = cms.untracked.string("")  #tilted barrel is assumed, use "flat" if running on flat
    )

TTTracksFromTrackletEmulation = cms.EDProducer("L1FPGATrackProducer",
                                               # general L1 tracking inputs
                                               SimTrackSource = cms.InputTag("g4SimHits"),
                                               SimVertexSource = cms.InputTag("g4SimHits"),
                                               TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                               asciiFileName = cms.untracked.string(""),
                                               failscenario = cms.untracked.int32(0),
                                               trackerGeometryType  = cms.untracked.string(""),  #tilted barrel is assumed, use "flat" if running on flat
                                               # specific emulation inputs 
                                               # (if running on CRAB use "../../fitpattern.txt" etc instead)
                                               Extended=cms.untracked.bool(False),
                                               Hnpar=cms.untracked.int32(4),
                                               fitPatternFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/fitpattern.txt'),
                                               memoryModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/memorymodules_hourglass.dat'), #change it to extended if running on it
                                               processingModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/processingmodules_hourglass.dat'), #change it to extended if running on it
                                               wiresFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/wires_hourglass.dat'), #change it to extended if running on it
                                               DTCLinkFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/calcNumDTCLinks.txt'),
                                               moduleCablingFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/modules_T5v3_27SP_nonant_tracklet.dat')
    )

TTTracksFromExtendedTrackletEmulation = cms.EDProducer("L1FPGATrackProducer",
                                               # general L1 tracking inputs
                                               SimTrackSource = cms.InputTag("g4SimHits"),
                                               SimVertexSource = cms.InputTag("g4SimHits"),
                                               TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                               asciiFileName = cms.untracked.string(""),
                                               failscenario = cms.untracked.int32(0),
                                               trackerGeometryType  = cms.untracked.string(""),  #tilted barrel is assumed, use "flat" if running on flat
                                               # specific emulation inputs 
                                               # (if running on CRAB use "../../fitpattern.txt" etc instead)
                                               Extended=cms.untracked.bool(True),
                                               Hnpar=cms.untracked.int32(5),
                                               fitPatternFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/fitpattern.txt'),
                                               memoryModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/memorymodules_hourglassExtended.dat'), #change it to extended if running on it
                                               processingModulesFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/processingmodules_hourglassExtended.dat'), #change it to extended if running on it
                                               wiresFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/wires_hourglassExtended.dat'), #change it to extended if running on it
                                               DTCLinkFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/calcNumDTCLinks.txt'),
                                               moduleCablingFile  = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/modules_T5v3_27SP_nonant_tracklet.dat')
    )
