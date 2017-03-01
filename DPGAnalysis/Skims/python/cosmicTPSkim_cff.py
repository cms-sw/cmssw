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
                                             SALabel = cms.InputTag("cosmicMuons"),
                                             PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                             radius = cms.double(90.0),
                                             maxZ = cms.double(130.0),
)

cosmicMuonsEndCapsOnlyTkFilter       = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("cosmicMuonsEndCapsOnly"))
cosmicMuonsTkFilter                  = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("cosmicMuons"))
cosmicMuons1LegTkFilter              = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("cosmicMuons1Leg"))
globalCosmicMuonsBarrelOnlyTkFilter  = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("globalCosmicMuons"))
globalCosmicMuonsEndCapsOnlyTkFilter = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("globalCosmicMuons"))
globalCosmicMuonsTkFilter            = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("globalCosmicMuons"))
globalCosmicMuons1LegTkFilter        = cosmicMuonsBarrelOnlyTkFilter.clone(SALabel = cms.InputTag("globalCosmicMuons1Leg"))

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


