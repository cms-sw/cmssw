import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for muon based alignment using beam-halo muons
ALCARECOMuAlBeamHaloHLT = copy.deepcopy(hltHighLevel)
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
import copy
from RecoLocalMuon.CSCSegment.cscSegments_cfi import *
cscSegmentsALCARECOBH = copy.deepcopy(cscSegments)
import copy
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
CosmicMuonSeedALCARECOBH = copy.deepcopy(CosmicMuonSeed)
import copy
from RecoMuon.CosmicMuonProducer.cosmicMuons_cfi import *
cosmicMuonsALCARECOBH = copy.deepcopy(cosmicMuons)
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
CosmicMuonSeedALCARECOBH.EnableDTMeasurement = False
CosmicMuonSeedALCARECOBH.CSCRecSegmentLabel = 'cscSegmentsALCARECOBH'
cosmicMuonsALCARECOBH.MuonSeedCollectionLabel = 'CosmicMuonSeed'
cosmicMuonsALCARECOBH.EnableDTMeasurement = False
cosmicMuonsALCARECOBH.CSCRecSegmentLabel = 'cscSegmentsALCARECOBH'

