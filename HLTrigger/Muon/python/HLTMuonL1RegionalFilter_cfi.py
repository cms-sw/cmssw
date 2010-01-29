import FWCore.ParameterSet.Config as cms

HLTMuonL1RegionalFilter = cms.EDFilter( "HLTMuonL1RegionalFilter",
                                CandTag = cms.InputTag( "hltL1extraParticles" ),
                                PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu20" ),
                                EtaBoundaries = cms.vdouble( -2.5, -1.6, 1.6, 2.5 ),
                                MinPts = cms.vdouble( 20, 20, 20 ),
                                QualityBitMasks = cms.vint32( 192, 128, 192 ),
                                MinN = cms.int32( 1 ),
                                SaveTag = cms.untracked.bool( True )
)

