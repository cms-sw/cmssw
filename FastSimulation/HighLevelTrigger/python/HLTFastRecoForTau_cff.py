import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.HLTPixelTracksProducer_cfi
hltRegionalPixelTracks = FastSimulation.Tracking.HLTPixelTracksProducer_cfi.hltPixelTracks.clone()
hltRegionalPixelTracks.FilterPSet.ptMin = 0.1
hltRegionalPixelTracks.RegionFactoryPSet.ComponentName = "L3MumuTrackingRegion"
hltRegionalPixelTracks.RegionFactoryPSet.RegionPSet = cms.PSet(
    originRadius = cms.double( 1.0 ),
    ptMin = cms.double( 0.5 ),
    originHalfLength = cms.double( 15.0 ),
    vertexZDefault = cms.double( 0.0 ),
    vertexSrc = cms.string( "hltDisplacedmumuVtxProducerTauTo2Mu" ),
    deltaEtaRegion = cms.double( 0.5 ),
    deltaPhiRegion = cms.double( 0.5 ),
    TrkSrc = cms.InputTag( "hltL3Muons" ),
    UseVtxTks = cms.bool( False )
)
