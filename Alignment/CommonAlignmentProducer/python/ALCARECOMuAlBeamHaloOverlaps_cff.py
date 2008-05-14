import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
ALCARECOMuAlBeamHaloOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
import RecoLocalMuon.CSCSegment.cscSegments_cfi
cscSegmentsALCARECOBHO = RecoLocalMuon.CSCSegment.cscSegments_cfi.cscSegments.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
CosmicMuonSeedALCARECOBHO = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsALCARECOBHO = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
ALCARECOMuAlBeamHaloOverlapsEnergyCut = cms.EDFilter("AlignmentCSCBeamHaloSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("cosmicMuonsALCARECOBHO"),
    minStations = cms.uint32(0), ## no "energy cut" yet

    minHitsPerStation = cms.uint32(4)
)

ALCARECOMuAlBeamHaloOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlBeamHaloOverlapsEnergyCut"),
    minHitsPerChamber = cms.uint32(4),
    station = cms.int32(0) ## all stations: I'll need to split it by station (8 subsamples) offline

)

reconstructAsCosmicMuonsALCARECOBHO = cms.Sequence(cscSegmentsALCARECOBHO*CosmicMuonSeedALCARECOBHO*cosmicMuonsALCARECOBHO)
seqALCARECOMuAlBeamHaloOverlaps = cms.Sequence(ALCARECOMuAlBeamHaloOverlapsHLT+reconstructAsCosmicMuonsALCARECOBHO*ALCARECOMuAlBeamHaloOverlapsEnergyCut*ALCARECOMuAlBeamHaloOverlaps)
ALCARECOMuAlBeamHaloOverlapsHLT.HLTPaths = ['CandHLTCSCBeamHaloOverlapRing1', 'CandHLTCSCBeamHaloOverlapRing2']
# Choice of the building algo: 1 SK, 2 TC, 3 DF, 4 ST, ...
cscSegmentsALCARECOBHO.algo_type = 4
cscSegmentsALCARECOBHO.inputObjects = 'hltCsc2DRecHits'
CosmicMuonSeedALCARECOBHO.EnableDTMeasurement = False
CosmicMuonSeedALCARECOBHO.CSCRecSegmentLabel = 'cscSegmentsALCARECOBHO'
cosmicMuonsALCARECOBHO.MuonSeedCollectionLabel = 'CosmicMuonSeedALCARECOBHO'
cosmicMuonsALCARECOBHO.TrajectoryBuilderParameters.EnableDTMeasurement = False
cosmicMuonsALCARECOBHO.TrajectoryBuilderParameters.CSCRecSegmentLabel = 'cscSegmentsALCARECOBHO'

