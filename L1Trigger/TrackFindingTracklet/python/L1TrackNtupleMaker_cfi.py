import FWCore.ParameterSet.Config as cms

# Configuration for Ntuple maker for analyzing L1 track performance.
# The parameters specified here are suitable for  Hybrid prompt track collections

L1TrackNtupleMaker = cms.EDAnalyzer('L1TrackNtupleMaker',
      # MyProcess is the (unsigned) PDGID corresponding to the process which is run
      # e.g. single electron/positron = 11
      #      single pion+/pion- = 211
      #      single muon+/muon- = 13
      #      pions in jets = 6
      #      taus = 15
      #      all TPs = 1 (pp collisions)
       MyProcess = cms.int32(1),
       DebugMode = cms.bool(False),      # printout lots of debug statements
       SaveAllTracks = cms.bool(True),   # save *all* L1 tracks, not just truth matched to primary particle
       SaveStubs = cms.bool(False),      # save some info for *all* stubs
       L1Tk_nPar = cms.int32(4), # use 4 or 5-parameter L1 tracking?
       L1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
       TP_minNStub = cms.int32(4),       # require TP to have >= X number of stubs associated with it
       TP_minNStubLayer = cms.int32(4),  # require TP to have stubs in >= X layers/disks
       TP_minPt = cms.double(1.9),       # only save TPs with pt > X GeV
       TP_maxEta = cms.double(2.5),      # only save TPs with |eta| < X
       TP_maxZ0 = cms.double(30.0),      # only save TPs with |z0| < X cm
       L1TrackInputTag = cms.InputTag("l1tTTTracksFromTrackletEmulation",  "Level1TTTracks"),         # TTTrack input
       MCTruthTrackInputTag = cms.InputTag( "TTTrackAssociatorFromPixelDigis",  "Level1TTTracks"),  # MCTruth input
       # other input collections
       L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
       MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
       MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
       TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
       TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
       # tracking in jets (--> requires AK4 genjet collection present!)
       TrackingInJets = cms.bool(False),
       GenJetInputTag = cms.InputTag("ak4GenJets", "")
)

