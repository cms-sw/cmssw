import FWCore.ParameterSet.Config as cms

# producer for alcaisotrk (HCAL isolated tracks)
IsoProd = cms.EDProducer("AlCaIsoTracksProducer",
                         TrackLabel        = cms.InputTag("generalTracks"),
                         VertexLabel       = cms.InputTag("offlinePrimaryVertices"),
                         BeamSpotLabel     = cms.InputTag("offlineBeamSpot"),
                         EBRecHitLabel     = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                         EERecHitLabel     = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                         HBHERecHitLabel   = cms.InputTag("hbhereco"),
                         L1GTSeedLabel     = cms.InputTag("hltL1sV0SingleJet60"),
                         TriggerEventLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                         TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                         IsoTrackLabel     = cms.string("HcalIsolatedTrackCollection"),
                         Triggers          = cms.vstring("HLT_IsoTrackHB","HLT_IsoTrackHE"),
                         ProcessName       = cms.string("HLT"),
# following 10 parameters are parameters to select good tracks
                         TrackQuality      = cms.string("highPurity"),
                         MinTrackPt        = cms.double(1.0),
                         MaxDxyPV          = cms.double(10.0),
                         MaxDzPV           = cms.double(100.0),
                         MaxChi2           = cms.double(5.0),
                         MaxDpOverP        = cms.double(0.1),
                         MinOuterHit       = cms.int32(4),
                         MinLayerCrossed   = cms.int32(8),
                         MaxInMiss         = cms.int32(2),
                         MaxOutMiss        = cms.int32(2),
# Minimum momentum of selected isolated track and signal zone
                         ConeRadius        = cms.double(34.98),
                         MinimumTrackP     = cms.double(20.0),
# signal zone in ECAL and MIP energy cutoff
                         ConeRadiusMIP     = cms.double(14.0),
                         MaximumEcalEnergy = cms.double(2.0),
# following 3 parameters are for isolation cuts and described in the code
                         MaxTrackP         = cms.double(8.0),
                         SlopeTrackP       = cms.double(0.05090504066),
                         IsolationEnergy   = cms.double(10.0),
# Prescale events only containing isolated tracks in the range
                         MomentumRangeLow  = cms.double(20.0),
                         MomentumRangeHigh = cms.double(40.0),
                         PreScaleFactor    = cms.int32(10)
)
