import FWCore.ParameterSet.Config as cms
from PhysicsTools.StarterKit.kinAxis_cfi  import *


patAnalyzerKit = cms.EDProducer("PatAnalyzerKit",
    ntuplize = cms.string('all'),
    outputTextName = cms.string('PatAnalyzerKit_output.txt'),
    enable = cms.string(''),
    disable = cms.string(''),
    muonAxis     = kinAxis(0, 200, 0, 200),
    electronAxis = kinAxis(0, 200, 0, 200),
    tauAxis      = kinAxis(0, 200, 0, 200),
    jetAxis      = kinAxis(0, 200, 0, 200),
    METAxis      = kinAxis(0, 200, 0, 200),
    photonAxis   = kinAxis(0, 200, 0, 200),
    trackAxis    = kinAxis(0, 200, 0, 200)
)


