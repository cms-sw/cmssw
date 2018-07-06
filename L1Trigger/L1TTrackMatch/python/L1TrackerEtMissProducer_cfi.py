import FWCore.ParameterSet.Config as cms

L1TrackerEtMiss = cms.EDProducer('L1TrackerEtMissProducer',
     L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
     maxZ0 = cms.double ( 25. ) ,    # in cm
     chi2Max = cms.double( 100. ),
     minPt = cms.double( 2. ),       # in GeV
     DeltaZ = cms.double( 1. ),      # in cm
     nStubsmin = cms.int32( 4 ),     # min number of stubs for the tracks to enter in TrkMET calculation
     nStubsPSmin = cms.int32( 0 ),   # min number of stubs in the PS Modules
     maxPt = cms.double( 50. ),	     # in GeV. When maxPt > 0, tracks with PT above maxPt are considered as
                                     # mismeasured and are treated according to HighPtTracks below.
                                     # When maxPt < 0, no special treatment is done for high PT tracks.
     HighPtTracks = cms.int32( 0 ),  # when = 0 : truncation. Tracks with PT above maxPt are ignored
                                     # when = 1 : saturation. Tracks with PT above maxPt are set to PT=maxPt.
                                     # When maxPt < 0, no special treatment is done for high PT tracks.
     doPtComp = cms.bool( False ),   # track-stubs PT compatibility cut
     doTightChi2 = cms.bool( False ) # chi2dof < 5 for tracks with PT > 10
)
