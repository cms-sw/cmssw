import FWCore.ParameterSet.Config as cms
from PhysicsTools.StarterKit.kinAxis_cfi  import *


CompositeKitDemo = cms.EDProducer("CompositeKit",
    description = cms.string('Higgs to Z + Z'),
    outputTextName = cms.string('CompositeKitKit_output.txt'),
    enable = cms.string(''),
    disable = cms.string(''),
    src = cms.InputTag("hToZZ"),
    ntuplize = cms.string('all'),
    electronSrc = cms.InputTag("selectedLayer1Electrons"),
    tauSrc = cms.InputTag("selectedLayer1Taus"),
    muonSrc = cms.InputTag("selectedLayer1Muons"),
    jetSrc = cms.InputTag("selectedLayer1Jets"),
    photonSrc = cms.InputTag("selectedLayer1Photons"),
    METSrc = cms.InputTag("selectedLayer1METs"),
    muonAxis     = kinAxis(0, 200, 0, 200),
    electronAxis = kinAxis(0, 200, 0, 200),
    tauAxis      = kinAxis(0, 200, 0, 200),
    jetAxis      = kinAxis(0, 200, 0, 200),
    METAxis      = kinAxis(0, 200, 0, 200),
    photonAxis   = kinAxis(0, 200, 0, 200),
    trackAxis    = kinAxis(0, 200, 0, 200),
    compositeAxis= kinAxis(0, 100, 0, 300)
)


