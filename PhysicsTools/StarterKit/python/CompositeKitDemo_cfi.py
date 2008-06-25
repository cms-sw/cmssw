import FWCore.ParameterSet.Config as cms

CompositeKitDemo = cms.EDProducer("CompositeKit",
    description = cms.string('Higgs to Z + Z'),
    outputTextName = cms.string('CompositeKitKit_output.txt'),
    enable = cms.string(''),
    disable = cms.string(''),
    src = cms.InputTag("hToZZ"),
    ntuplize = cms.string('all'),
    m1 = cms.double(0.0),
    m2 = cms.double(200.0),
    pt2 = cms.double(200.0),
    pt1 = cms.double(0.0),
    resonanceM1 = cms.double(0.0),
    resonanceM2 = cms.double(400.0)
)


