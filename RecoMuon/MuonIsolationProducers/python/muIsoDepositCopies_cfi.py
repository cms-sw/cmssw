import FWCore.ParameterSet.Config as cms

muIsoDepositTk = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons1stStep:tracker")),
  depositNames = cms.vstring('')
)

muIsoDepositJets = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons1stStep:jets")),
  depositNames = cms.vstring('')
)

muIsoDepositCalByAssociatorTowers = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons1stStep:ecal"), cms.InputTag("muons1stStep:hcal"), cms.InputTag("muons1stStep:ho")),
  depositNames = cms.vstring('ecal', 'hcal', 'ho')
)


