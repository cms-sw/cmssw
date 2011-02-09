import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

cosmictrackfinderP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                              src = cms.InputTag('cosmictrackfinderP5'),
                                              minNumber = cms.uint32(1) 
                                              )

ctfWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                  src = cms.InputTag('ctfWithMaterialTracksP5'),
                                                  minNumber = cms.uint32(1) 
                                                  )

rsWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                 src = cms.InputTag('rsWithMaterialTracksP5'),
                                                 minNumber = cms.uint32(1) 
                                                 )

cosmicMuonsBarrelOnlyTkFilter = cms.EDFilter("HLTMuonPointingFilter",
                                             SALabel = cms.string("cosmicMuonsBarrelOnly"),
                                             PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                             radius = cms.double(90.0),
                                             maxZ = cms.double(130.0)
                                             )

cosmicMuonsEndCapsOnlyTkFilter       = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("cosmicMuonsEndCapsOnly"))
cosmicMuonsTkFilter                  = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("cosmicMuons"))
cosmicMuons1LegTkFilter              = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("cosmicMuons1Leg"))
globalCosmicMuonsBarrelOnlyTkFilter  = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("globalCosmicMuonsBarrelOnly"))
globalCosmicMuonsEndCapsOnlyTkFilter = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("globalCosmicMuonsEndCapsOnly"))
globalCosmicMuonsTkFilter            = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("globalCosmicMuons"))
globalCosmicMuons1LegTkFilter        = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.string("globalCosmicMuons1Leg"))

cosmicMuonsBarrelOnlyTkSequence        = cms.Sequence(cosmicMuonsBarrelOnlyTkFilter)
cosmicMuonsEndCapsOnlyTkSequence       = cms.Sequence(cosmicMuonsEndCapsOnlyTkFilter)
cosmicMuonsTkSequence                  = cms.Sequence(cosmicMuonsTkFilter)
cosmicMuons1LegTkSequence              = cms.Sequence(cosmicMuons1LegTkFilter)
globalCosmicMuonsBarrelOnlyTkSequence  = cms.Sequence(globalCosmicMuonsBarrelOnlyTkFilter)
globalCosmicMuonsEndCapsOnlyTkSequence = cms.Sequence(globalCosmicMuonsEndCapsOnlyTkFilter)
globalCosmicMuonsTkSequence            = cms.Sequence(globalCosmicMuonsTkFilter)
globalCosmicMuons1LegTkSequence        = cms.Sequence(globalCosmicMuons1LegTkFilter)
cosmictrackfinderP5TkCntSequence       = cms.Sequence(cosmictrackfinderP5TkCntFilter)
ctfWithMaterialTracksP5TkCntSequence   = cms.Sequence(ctfWithMaterialTracksP5TkCntFilter)
rsWithMaterialTracksP5TkCntSequence    = cms.Sequence(rsWithMaterialTracksP5TkCntFilter)


