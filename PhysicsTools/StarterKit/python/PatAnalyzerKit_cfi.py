import FWCore.ParameterSet.Config as cms
from PhysicsTools.StarterKit.kinAxis_cfi  import *


patAnalyzerKit = cms.EDProducer("PatAnalyzerKit",
    ntuplize = cms.string('all'),
    outputTextName = cms.string('PatAnalyzerKit_output.txt'),
    enable = cms.string(''),
    disable = cms.string(''),
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
    trackAxis    = kinAxis(0, 200, 0, 200)
)


