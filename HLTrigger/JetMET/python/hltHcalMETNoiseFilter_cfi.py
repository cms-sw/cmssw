import FWCore.ParameterSet.Config as cms

hltHcalMETNoiseFilter = cms.EDFilter("HLTHcalMETNoiseFilter",
    HcalNoiseRBXCollection = cms.InputTag("hcalnoise"),
    HcalNoiseSummary       = cms.InputTag("hcalnoise"),
    severity = cms.int32(1),
    EMFractionMin = cms.double(0.1),
    nRBXhitsMax = cms.int32(50),
    RBXhitThresh = cms.double(1.5)
)
