import FWCore.ParameterSet.Config as cms

LowPtTrimuonPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
    ),
    recMuonLabel  = cms.InputTag("muons"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(3),
    # -- Analysis specific binnings
    parametersDxy      = cms.vdouble(50, -2.500, 2.500),
    )
