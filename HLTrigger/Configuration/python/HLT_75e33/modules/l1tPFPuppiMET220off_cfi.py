import FWCore.ParameterSet.Config as cms

l1tPFPuppiMET220off = cms.EDFilter("L1TEnergySumFilter",
    MinPt = cms.double(220.0),
    Scalings = cms.PSet(
        theScalings = cms.vdouble(54.2859, 1.39739, 0)
    ),
    TypeOfSum = cms.string('MET'),
    inputTag = cms.InputTag("l1tMETPFProducer")
)
