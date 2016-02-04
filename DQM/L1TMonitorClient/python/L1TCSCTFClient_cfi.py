import FWCore.ParameterSet.Config as cms

l1tcsctfClient = cms.EDAnalyzer("L1TCSCTFClient",
    input_dir = cms.untracked.string('L1T/L1TCSCTF'),
    prescaleLS = cms.untracked.int32(-1),
    verbose = cms.untracked.bool(False),
    prescaleEvt = cms.untracked.int32(500),
    output_dir = cms.untracked.string('L1T/L1TCSCTF/QualityTests')
)


