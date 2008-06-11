import FWCore.ParameterSet.Config as cms

CompositeKitDemo = cms.EDFilter("CompositeKit",
    enable = cms.string(''),
    description = cms.string('Higgs to Z + Z'),
    outputTextName = cms.string('CompositeKitKit_output.txt'),
    resonanceM1 = cms.double(0.0),
    resonanceM2 = cms.double(200.0),
    disable = cms.string(''),
    m1 = cms.double(0.0),
    compositeCandTag = cms.InputTag("hToZZ"),
    isCSA07Soup = cms.bool(False),
    m2 = cms.double(200.0),
    ntuplize = cms.string('all'),
    pt2 = cms.double(200.0),
    pt1 = cms.double(0.0)
)


