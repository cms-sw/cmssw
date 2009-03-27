import FWCore.ParameterSet.Config as cms

#============================================================
# the RPCNoise filter
#============================================================
rpcNoise = cms.EDFilter(
    "RPCNoise",
    fillHistograms = cms.untracked.bool(True),
    histogramFileName = cms.untracked.string('histos_test.root'),
    nRPCHitsCut  = cms.untracked.int32(40),
    nCSCWiresCut  = cms.untracked.int32(10),
    nCSCStripsCut  = cms.untracked.int32(50),
    nDTDigisCut  = cms.untracked.int32(40)
)
