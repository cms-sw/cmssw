import FWCore.ParameterSet.Config as cms

muIsoDepositTk = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons:tracker")),
  depositNames = cms.vstring('')
)

muIsoDepositJets = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons:jets")),
  depositNames = cms.vstring('')
)

muIsoDepositCalByAssociatorTowers = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons:ecal"), cms.InputTag("muons:hcal"), cms.InputTag("muons:ho")),
  depositNames = cms.vstring('ecal', 'hcal', 'ho')
)


