import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

cosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                   SALabel = cms.string("cosmicMuonsBarrelOnly"),
                                                   PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                   radius = cms.double(10.0),
                                                   maxZ = cms.double(50.0)
                                                   )

cosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                         SALabel = cms.string("cosmicMuons"),
                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                         radius = cms.double(10.0),
                                         maxZ = cms.double(50.0)
                                         )

cosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                       SALabel = cms.string("cosmicMuons1Leg"),
                                                       PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                       radius = cms.double(10.0),
                                                       maxZ = cms.double(50.0)
                                                       )

globalCosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                         SALabel = cms.string("globalCosmicMuonsBarrelOnly"),
                                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                         radius = cms.double(10.0),
                                                         maxZ = cms.double(50.0)
                                                         )

cosmictrackfinderP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                 SALabel = cms.string("cosmictrackfinderP5"),
                                                 PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                 radius = cms.double(10.0),
                                                 maxZ = cms.double(50.0)
                                                 )

globalCosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                               SALabel = cms.string("globalCosmicMuons"),
                                               PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                               radius = cms.double(10.0),
                                               maxZ = cms.double(50.0)
                                               )

rsWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                    SALabel = cms.string("rsWithMaterialTracksP5"),
                                                    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                    radius = cms.double(10.0),
                                                    maxZ = cms.double(50.0)
                                                    )

globalCosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                             SALabel = cms.string("globalCosmicMuons1Leg"),
                                                             PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                             radius = cms.double(10.0),
                                                             maxZ = cms.double(50.0)
                                                             )

ctfWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.string("ctfWithMaterialTracksP5"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(10.0),
                                                     maxZ = cms.double(50.0)
                                                     )


cosmicMuonsBarrelOnlySequence = cms.Sequence(cosmicMuonsBarrelOnlyFilter)
cosmicMuonsSequence = cms.Sequence(cosmicMuonsFilter)
cosmicMuons1LegSequence = cms.Sequence(cosmicMuons1LegFilter)
globalCosmicMuonsBarrelOnlySequence = cms.Sequence(globalCosmicMuonsBarrelOnlyFilter)
cosmictrackfinderP5Sequence = cms.Sequence(cosmictrackfinderP5Filter)
globalCosmicMuonsSequence = cms.Sequence(globalCosmicMuonsFilter)
rsWithMaterialTracksP5Sequence = cms.Sequence(rsWithMaterialTracksP5Filter)
globalCosmicMuons1LegSequence = cms.Sequence(globalCosmicMuons1LegFilter)
ctfWithMaterialTracksP5Sequence = cms.Sequence(ctfWithMaterialTracksP5Filter)


