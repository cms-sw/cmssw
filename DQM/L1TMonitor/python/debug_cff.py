import FWCore.ParameterSet.Config as cms

d0 = cms.EDAnalyzer("EventContentAnalyzer",
    verbose = cms.untracked.bool(False)
)

d1 = cms.EDAnalyzer("DumpFEDRawDataProduct",
    #untracked vint32 feds = { 745 }
    dumpPayload = cms.untracked.bool(False)
)

debugpath = cms.Path(d0+d1)

