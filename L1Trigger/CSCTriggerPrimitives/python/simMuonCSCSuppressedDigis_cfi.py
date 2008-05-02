process.simMuonCSCSuppressedDigis = cms.EDProducer("CSCDigiSuppressor",
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    lctTag = cms.InputTag("cscTriggerPrimitiveDigis")
)

