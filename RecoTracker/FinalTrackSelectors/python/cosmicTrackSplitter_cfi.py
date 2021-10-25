import FWCore.ParameterSet.Config as cms

cosmicTrackSplitter = cms.EDProducer("CosmicTrackSplitter",
                                     stripFrontInvalidHits = cms.bool(True),
                                     stripBackInvalidHits = cms.bool(True),
                                     stripAllInvalidHits = cms.bool(False),
                                     replaceWithInactiveHits = cms.bool(False),
                                     excludePixelHits = cms.bool(False),
                                     tracks = cms.InputTag("cosmictrackfinderP5"),
                                     tjTkAssociationMapTag = cms.InputTag("cosmictrackfinderP5"),
                                     minimumHits = cms.uint32(6),
                                     detsToIgnore = cms.vuint32(),
                                     dzCut = cms.double( 9999.0 ),
                                     dxyCut = cms.double( 9999.0 )
                                     )

