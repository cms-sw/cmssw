import FWCore.ParameterSet.Config as cms

HLTMuonIsoFilter = cms.EDFilter( "HLTMuonIsoFilter",
                                 CandTag = cms.InputTag( "hltL3MuonCandidates" ),
                                 PreviousCandTag = cms.InputTag( '' ),
                                 MinN = cms.int32( 1 ),
                                 DepTag = cms.VInputTag( 'hltL3MuonIsolations' ),
                                 IsolatorPSet = cms.PSet(  ),
                                 saveTags = cms.bool( False )
                                 )
