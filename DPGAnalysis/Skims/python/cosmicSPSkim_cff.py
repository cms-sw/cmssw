import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

cosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("cosmicMuons"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

cosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("cosmicMuons"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

cosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("cosmicMuons1Leg"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

globalCosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("globalCosmicMuons"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

cosmictrackfinderP5Filter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("cosmictrackfinderP5"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

globalCosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("globalCosmicMuons"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

rsWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("rsWithMaterialTracksP5"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

globalCosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("globalCosmicMuons1Leg"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.0),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
)

ctfWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
    SALabel = cms.InputTag("ctfWithMaterialTracksP5"),
    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
    radius = cms.double(10.0),
    maxZ = cms.double(50.00),
    PixHits = cms.uint32(0),
    TkLayers = cms.uint32(0),
    MuonHits = cms.uint32(0),                              
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


