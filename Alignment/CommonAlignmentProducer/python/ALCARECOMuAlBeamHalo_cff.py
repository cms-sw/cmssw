import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for muon based alignment using beam-halo muons
ALCARECOMuAlBeamHaloHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
import RecoLocalMuon.CSCSegment.cscSegments_cfi
cscSegmentsALCARECOBH = RecoLocalMuon.CSCSegment.cscSegments_cfi.cscSegments.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
CosmicMuonSeedALCARECOBH = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsALCARECOBH = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
ALCARECOMuAlBeamHalo = cms.EDFilter("AlignmentCSCBeamHaloSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("cosmicMuonsALCARECOBH"),
    minStations = cms.uint32(0), ## no "energy cut" yet

    minHitsPerStation = cms.uint32(4)
)

reconstructAsCosmicMuonsALCARECOBH = cms.Sequence(cscSegmentsALCARECOBH*CosmicMuonSeedALCARECOBH*cosmicMuonsALCARECOBH)
seqALCARECOMuAlBeamHalo = cms.Sequence(ALCARECOMuAlBeamHaloHLT+reconstructAsCosmicMuonsALCARECOBH*ALCARECOMuAlBeamHalo)
ALCARECOMuAlBeamHaloHLT.HLTPaths = ['CandHLTCSCBeamHalo', 'CandHLTCSCBeamHaloRing2or3']
# Choice of the building algo: 1 SK, 2 TC, 3 DF, 4 ST, ...
cscSegmentsALCARECOBH.algo_type = 4
cscSegmentsALCARECOBH.inputObjects = 'hltCsc2DRecHits'
CosmicMuonSeedALCARECOBH.EnableDTMeasurement = False
CosmicMuonSeedALCARECOBH.CSCRecSegmentLabel = 'cscSegmentsALCARECOBH'
cosmicMuonsALCARECOBH.MuonSeedCollectionLabel = 'CosmicMuonSeedALCARECOBH'
# replace cosmicMuonsALCARECOBH.EnableDTMeasurement = false
cosmicMuonsALCARECOBH.TrajectoryBuilderParameters.EnableDTMeasurement = False
cosmicMuonsALCARECOBH.TrajectoryBuilderParameters.CSCRecSegmentLabel = 'cscSegmentsALCARECOBH'

