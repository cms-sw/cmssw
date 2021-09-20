import FWCore.ParameterSet.Config as cms

pSetPvClusterComparerForIT = cms.PSet(
    track_chi2_max = cms.double(20.0),
    track_prob_min = cms.double(-1.0),
    track_pt_max = cms.double(30.0),        # this should 100. according to the Phase 2 HLT Tracking instructions
    track_pt_min = cms.double(1.0)
)
