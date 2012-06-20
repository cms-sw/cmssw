import FWCore.ParameterSet.Config as cms

HLTMuonL3PreFilter = cms.EDFilter( "HLTMuonL3PreFilter",
                                   BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
                                   CandTag = cms.InputTag( "hltL3MuonCandidates" ),
                                   PreviousCandTag = cms.InputTag( '' ),
                                   MinN = cms.int32( 1 ),
                                   MaxEta = cms.double( 2.5 ),
                                   MinNhits = cms.int32( 0 ),
                                   MaxDr = cms.double( 2.0 ),
                                   MaxDz = cms.double( 9999.0 ),
                                   MinPt = cms.double( 3.0 ),
                                   NSigmaPt = cms.double( 0.0 ),
                                   SaveTag = cms.untracked.bool( False )
                                   )
