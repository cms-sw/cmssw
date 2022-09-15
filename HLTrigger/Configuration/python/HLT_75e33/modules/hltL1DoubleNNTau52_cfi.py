import FWCore.ParameterSet.Config as cms

hltL1DoubleNNTau52 = cms.EDFilter("L1TPFTauFilter",
    MaxEta = cms.double(2.172),
    MinEta = cms.double(-2.172),
    MinN = cms.int32(2),
    MinPt = cms.double(52.0),
    PassLooseNN = cms.int32(0),
    Scalings = cms.PSet(
        barrel = cms.vdouble(-9.54135, 1.73403, 0),
        endcap = cms.vdouble(-36.157, 3.83749, 0)
    ),
    inputTag = cms.InputTag("l1tNNTauProducerPuppi","L1PFTausNN"),
    saveTags = cms.bool(True)
)
