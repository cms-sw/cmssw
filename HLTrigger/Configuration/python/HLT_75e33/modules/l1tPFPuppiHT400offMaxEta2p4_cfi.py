import FWCore.ParameterSet.Config as cms

l1tPFPuppiHT400offMaxEta2p4 = cms.EDFilter("L1TEnergySumFilter",
    MinPt = cms.double(400.0),
    Scalings = cms.PSet(
        theScalings = cms.vdouble(50.0182, 1.0961, 0)
    ),
    TypeOfSum = cms.string('HT'),
    inputTag = cms.InputTag("l1tPFPuppiHT")
)
