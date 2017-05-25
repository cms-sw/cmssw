import FWCore.ParameterSet.Config as cms

ctppsPixelDQMSource = cms.EDAnalyzer("CTPPSPixelDQMSource",
#    tagRPixDigi = cms.InputTag("RPixDetDigitizer", ""),
    tagRPixDigi = cms.InputTag("ctppsPixelDigis", ""),
    tagRPixCluster = cms.InputTag("clusterProd", ""),  
    RPStatusWord = cms.untracked.uint32(0x8000), # rpots included in readout
    verbosity = cms.untracked.uint32(0)
)
