import FWCore.ParameterSet.Config as cms

l1tTrackerEmuHTMiss = cms.EDProducer("L1TkHTMissEmulatorProducer",
    L1TkJetEmulationInputTag = cms.InputTag("l1tTrackJetsEmulation", "L1TrackJets"),
    L1MHTCollectionName = cms.string("L1TrackerEmuHTMiss"),
    jet_maxEta = cms.double(2.4),
    jet_minPt = cms.double(5.0),
    jet_minNtracksLowPt = cms.int32(2),
    jet_minNtracksHighPt = cms.int32(3),
    debug = cms.bool(False),
    displaced = cms.bool(False)
)

l1tTrackerEmuHTMissExtended = cms.EDProducer("L1TkHTMissEmulatorProducer",
    L1TkJetEmulationInputTag = cms.InputTag("l1tTrackJetsExtendedEmulation", "L1TrackJetsExtended"),
    L1MHTCollectionName = cms.string("L1TrackerEmuHTMissExtended"),
    jet_maxEta = cms.double(2.4),
    jet_minPt = cms.double(5.0),
    jet_minNtracksLowPt = cms.int32(2),
    jet_minNtracksHighPt = cms.int32(3),
    debug = cms.bool(False),
    displaced = cms.bool(True)
)
