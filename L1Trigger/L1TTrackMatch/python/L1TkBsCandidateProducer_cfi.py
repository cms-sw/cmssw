import FWCore.ParameterSet.Config as cms

# Candidates with Medium WP (default)
L1TkBsCandidates = cms.EDProducer("L1TkBsCandidateProducer", 
  verbose           = cms.bool(False),
  L1TrackInputTag   = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
  TrackEtaMax       = cms.double(2.5),
  TrackPtMin        = cms.double(2.0), # GeV
  ApplyTrackQuality = cms.bool(False),
  TrackChi2Max      = cms.double(20.0),
  TrackLayersMin    = cms.int32(4),
  TrackPSLayersMin  = cms.int32(2),
  TrackPairDzMax    = cms.double(1.0), # cm
  TrackPairDxyMax   = cms.double(1.0), # cm
  PhiMassMin        = cms.double(1.0), # GeV
  PhiMassMax        = cms.double(1.03), # GeV
  PhiPairDzMax      = cms.double(1.0),
  PhiPairDxyMax     = cms.double(1.0),
  PhiPairDrMin      = cms.double(0.2),
  PhiPairDrMax      = cms.double(1.0),
  PhiTrkPairDrMax   = cms.double(0.12),
  BsMassMin         = cms.double(5.29), # GeV
  BsMassMax         = cms.double(5.48), # GeV
  label             = cms.string("")
)
# Candidates with Loose WP
L1TkBsCandidatesLooseWP = L1TkBsCandidates.clone()
L1TkBsCandidatesLooseWP.PhiMassMin = cms.double(0.99) # GeV
L1TkBsCandidatesLooseWP.PhiMassMax = cms.double(1.04)
L1TkBsCandidatesLooseWP.BsMassMin  = cms.double(5.27) # GeV
L1TkBsCandidatesLooseWP.BsMassMax  = cms.double(5.49) # GeV

# Candidates with Tight WP
L1TkBsCandidatesTightWP = L1TkBsCandidates.clone()
L1TkBsCandidatesTightWP.TrackPairDzMax  = cms.double(0.3) # cm
L1TkBsCandidatesTightWP.TrackPairDxyMax = cms.double(0.5) # cm
L1TkBsCandidatesTightWP.PhiPairDzMax    = cms.double(1.0)
L1TkBsCandidatesTightWP.PhiPairDxyMax   = cms.double(0.5)

