import FWCore.ParameterSet.Config as cms

l1tTrackerEmuHTMissExtended = cms.EDProducer("L1TkHTMissEmulatorProducer",
    L1MHTCollectionName = cms.string('L1TrackerEmuHTMissExtended'),
    L1TkJetEmulationInputTag = cms.InputTag("l1tTrackJetsExtendedEmulation","L1TrackJetsExtended"),
    debug = cms.bool(False),
    displaced = cms.bool(True),
    jet_maxEta = cms.double(2.4),
    jet_minNtracksHighPt = cms.int32(0),
    jet_minNtracksLowPt = cms.int32(0),
    jet_minPt = cms.double(3.0)
)
