import FWCore.ParameterSet.Config as cms

L1TrackerEtMiss = cms.EDProducer('L1TrackerEtMissProducer',
     L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
     L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"), #cms.InputTag("VertexProducer", "l1vertextdr"),

     maxZ0 = cms.double ( 15. ) ,    # in cm
     maxEta = cms.double ( 2.4 ) ,
     chi2dofMax = cms.double( 50. ),
     bendchi2Max = cms.double( 1.75 ),
     minPt = cms.double( 2. ),       # in GeV
     DeltaZ = cms.double( 3. ),      # in cm
     nStubsmin = cms.int32( 4 ),     # min number of stubs for the tracks to enter in TrkMET calculation
     nStubsPSmin = cms.int32( 2 ),   # min number of stubs in the PS Modules
     maxPt = cms.double( 200. ),	 # in GeV. When maxPt > 0, tracks with PT above maxPt are considered as
                                     # mismeasured and are treated according to HighPtTracks below.
                                     # When maxPt < 0, no special treatment is done for high PT tracks.
     HighPtTracks = cms.int32( 1 ),  # when = 0 : truncation. Tracks with PT above maxPt are ignored
                                     # when = 1 : saturation. Tracks with PT above maxPt are set to PT=maxPt.
                                     # When maxPt < 0, no special treatment is done for high PT tracks.
)
