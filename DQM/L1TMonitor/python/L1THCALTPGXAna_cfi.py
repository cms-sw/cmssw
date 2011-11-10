import FWCore.ParameterSet.Config as cms

l1thcaltpgxana = cms.EDFilter("L1THCALTPGXAna",
    hfrecoSource = cms.InputTag("hfreco"),
    fakeCut = cms.untracked.double(5.0),
    verbose = cms.untracked.bool(False),
    hbherecoSource = cms.InputTag("hbhereco"),
    DQMStore = cms.untracked.bool(True),
    hcaltpgSource = cms.InputTag("hcalDigis")
)


