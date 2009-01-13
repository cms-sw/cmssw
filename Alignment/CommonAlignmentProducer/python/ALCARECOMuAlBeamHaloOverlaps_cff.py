# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions

import FWCore.ParameterSet.Config as cms

# import HLTrigger.HLTfilters.hltHighLevel_cfi
# ALCARECOMuAlBeamHaloOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#     HLTPaths = ['HLT_CSCBeamHaloOverlapRing1', 'HLT_CSCBeamHaloOverlapRing2'],
#     throw = False # tolerate triggers stated above, but not available
#     )

# from RecoMuon.Configuration.RecoMuonCosmics_cff import *  # required as of 2_1_X

# import RecoLocalMuon.CSCSegment.cscSegments_cfi
# cscSegmentsALCARECOBHO = RecoLocalMuon.CSCSegment.cscSegments_cfi.cscSegments.clone()
# cscSegmentsALCARECOBHO.algo_type = 4 # Choice of the building algo: 1 SK, 2 TC, 3 DF, 4 ST, ...
# cscSegmentsALCARECOBHO.inputObjects = 'csc2DRecHits'

# import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
# CosmicMuonSeedALCARECOBHO = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
# CosmicMuonSeedALCARECOBHO.EnableDTMeasurement = False
# CosmicMuonSeedALCARECOBHO.CSCRecSegmentLabel = 'cscSegmentsALCARECOBHO'

# import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
# cosmicMuonsALCARECOBHO = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
# cosmicMuonsALCARECOBHO.MuonSeedCollectionLabel = 'CosmicMuonSeedALCARECOBHO'
# cosmicMuonsALCARECOBHO.TrajectoryBuilderParameters.EnableDTMeasurement = False
# cosmicMuonsALCARECOBHO.TrajectoryBuilderParameters.CSCRecSegmentLabel = 'cscSegmentsALCARECOBHO'

# reconstructAsCosmicMuonsALCARECOBHO = cms.Sequence(cscSegmentsALCARECOBHO * CosmicMuonSeedALCARECOBHO * cosmicMuonsALCARECOBHO)

ALCARECOMuAlBeamHaloOverlapsEnergyCut = cms.EDFilter("AlignmentCSCBeamHaloSelectorModule",
    filter = cms.bool(True),
#     src = cms.InputTag("cosmicMuonsALCARECOBHO"),
    src = cms.InputTag("cosmicMuons"), # get cosmicMuons from global-run reconstruction
    minStations = cms.uint32(0), # no "energy cut" yet
    minHitsPerStation = cms.uint32(4)
)

ALCARECOMuAlBeamHaloOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlBeamHaloOverlapsEnergyCut"),
    minHitsPerChamber = cms.uint32(4),
    station = cms.int32(0) # all stations: I'll need to split it by station (8 subsamples) offline
)

# seqALCARECOMuAlBeamHaloOverlaps = cms.Sequence(ALCARECOMuAlBeamHaloOverlapsHLT + reconstructAsCosmicMuonsALCARECOBHO * ALCARECOMuAlBeamHaloOverlapsEnergyCut * ALCARECOMuAlBeamHaloOverlaps)
# seqALCARECOMuAlBeamHaloOverlaps = cms.Sequence(reconstructAsCosmicMuonsALCARECOBHO * ALCARECOMuAlBeamHaloOverlapsEnergyCut * ALCARECOMuAlBeamHaloOverlaps)
seqALCARECOMuAlBeamHaloOverlaps = cms.Sequence(ALCARECOMuAlBeamHaloOverlapsEnergyCut * ALCARECOMuAlBeamHaloOverlaps)

