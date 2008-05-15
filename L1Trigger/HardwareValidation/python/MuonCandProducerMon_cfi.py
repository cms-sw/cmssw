import FWCore.ParameterSet.Config as cms

muonCandMon = cms.EDFilter("MuonCandProducerMon",
    DTinput = cms.untracked.InputTag("dttfdigis"),
    VerboseFlag = cms.untracked.int32(0),
    CSCinput = cms.untracked.InputTag("csctfdigis")
)


