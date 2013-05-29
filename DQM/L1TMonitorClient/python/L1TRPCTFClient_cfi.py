import FWCore.ParameterSet.Config as cms

l1trpctfClient = cms.EDAnalyzer("L1TRPCTFClient",
    input_dir = cms.untracked.string('L1T/L1TRPCTF'),
    prescaleEvt = cms.untracked.int32(1),
    verbose = cms.untracked.bool(False),
    output_dir = cms.untracked.string('L1T/L1TRPCTF/Client')
)


