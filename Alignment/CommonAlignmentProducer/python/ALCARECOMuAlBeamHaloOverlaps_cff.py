import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
ALCARECOMuAlBeamHaloOverlapsHLT = copy.deepcopy(hltHighLevel)
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
import copy
from RecoLocalMuon.CSCSegment.cscSegments_cfi import *
cscSegmentsALCARECOBHO = copy.deepcopy(cscSegments)
import copy
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
CosmicMuonSeedALCARECOBHO = copy.deepcopy(CosmicMuonSeed)
import copy
from RecoMuon.CosmicMuonProducer.cosmicMuons_cfi import *
cosmicMuonsALCARECOBHO = copy.deepcopy(cosmicMuons)
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
CosmicMuonSeedALCARECOBHO.EnableDTMeasurement = False
CosmicMuonSeedALCARECOBHO.CSCRecSegmentLabel = 'cscSegmentsALCARECOBHO'
cosmicMuonsALCARECOBHO.MuonSeedCollectionLabel = 'CosmicMuonSeed'
cosmicMuonsALCARECOBHO.EnableDTMeasurement = False
cosmicMuonsALCARECOBHO.CSCRecSegmentLabel = 'cscSegmentsALCARECOBHO'

