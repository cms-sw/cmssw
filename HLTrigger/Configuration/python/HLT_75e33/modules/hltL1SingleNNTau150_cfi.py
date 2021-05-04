import FWCore.ParameterSet.Config as cms

hltL1SingleNNTau150 = cms.EDFilter("L1TPFTauFilter",
    MaxEta = cms.double(2.172),
    MinEta = cms.double(-2.172),
    MinN = cms.int32(1),
    MinPt = cms.double(150.0),
    PassLooseNN = cms.int32(0),
    Scalings = cms.PSet(
        barrel = cms.vdouble(-9.54135, 1.73403, 0),
        endcap = cms.vdouble(-36.157, 3.83749, 0)
    ),
    inputTag = cms.InputTag("l1NNTauProducerPuppi","L1PFTausNN"),
    saveTags = cms.bool(True)
)
