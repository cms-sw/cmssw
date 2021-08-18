import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackTrigger.TrackQualityParams_cfi import *

TTTracksFromTrackletEmulation = cms.EDProducer("L1FPGATrackProducer",
                                               TTStubSource = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
                                               InputTagTTDTC = cms.InputTag("TrackerDTCProducer", "StubAccepted"), 
                                               readMoreMcTruth = cms.bool(True),
                                               MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
                                               MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                               TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
                                               BeamSpotSource = cms.InputTag("offlineBeamSpot"),
                                               asciiFileName = cms.untracked.string(""),
                                               Extended = cms.bool(False),
                                               Hnpar = cms.uint32(4),
                                               # (if running on CRAB use "../../fitpattern.txt" etc instead)
                                               fitPatternFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/fitpattern.txt'),
                                               memoryModulesFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/memorymodules_hourglassExtended.dat'),
                                               processingModulesFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/processingmodules_hourglassExtended.dat'),
                                               wiresFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/wires_hourglassExtended.dat'),
                                               # Quality Flag and Quality params
                                               TrackQuality = cms.bool(True),
                                               TrackQualityPSet = cms.PSet(TrackQualityParams)
    )

TTTracksFromExtendedTrackletEmulation = TTTracksFromTrackletEmulation.clone(
                                               Extended = cms.bool(True),
                                               Hnpar = cms.uint32(5),
                                               # specifying where the TrackletEngineDisplaced(TED)/TripletEngine(TRE) tables are located
                                               tableTEDFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/table_TED/table_TED_D1PHIA1_D2PHIA1.txt'),
                                               tableTREFile = cms.FileInPath('L1Trigger/TrackFindingTracklet/data/table_TRE/table_TRE_D1AD2A_1.txt'),
                                               # Quality Flag and Quality params
                                               TrackQuality = cms.bool(False),
                                               TrackQualityPSet = cms.PSet(TrackQualityParams)
    )

