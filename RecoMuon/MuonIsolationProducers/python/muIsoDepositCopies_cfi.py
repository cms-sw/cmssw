import FWCore.ParameterSet.Config as cms

muIsoDepositTk = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons1stStep:tracker")),
  depositNames = cms.vstring('')
)

muIsoDepositTkDisplaced = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("filteredDisplacedMuons1stStep:tracker")),
  depositNames = cms.vstring('')
)

muIsoDepositJets = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons1stStep:jets")),
  depositNames = cms.vstring('')
)

muIsoDepositJetsDisplaced = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("filteredDisplacedMuons1stStep:jets")),
  depositNames = cms.vstring('')
)

muIsoDepositCalByAssociatorTowers = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("muons1stStep:ecal"), cms.InputTag("muons1stStep:hcal"), cms.InputTag("muons1stStep:ho")),
  depositNames = cms.vstring('ecal', 'hcal', 'ho')
)

muIsoDepositCalByAssociatorTowersDisplaced = cms.EDProducer("MuIsoDepositCopyProducer",
  inputTags = cms.VInputTag(cms.InputTag("filteredDisplacedMuons1stStep:ecal"), cms.InputTag("filteredDisplacedMuons1stStep:hcal"), cms.InputTag("filteredDisplacedMuons1stStep:ho")),
  depositNames = cms.vstring('ecal', 'hcal', 'ho')
)

