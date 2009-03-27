import FWCore.ParameterSet.Config as cms

cosmicTrackSplitter = cms.EDFilter("CosmicTrackSplitter",
                                   stripFrontInvalidHits = cms.bool(True),
                                   stripBackInvalidHits = cms.bool(True),
                                   stripAllInvalidHits = cms.bool(False),
                                   replaceWithInactiveHits = cms.bool(False),
                                   tracks = cms.InputTag("TrackRefitterP5"),
                                   tjTkAssociationMapTag = cms.InputTag("TrackRefitterP5"),
                                   minimumHits = cms.uint32(6),
                                   detsToIgnore = cms.vuint32(),
                                   dzCut = cms.double( 9999.0 ),
                                   dxyCut = cms.double( 9999.0 )
                                   )

