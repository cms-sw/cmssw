import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *

## for scheduled mode
makePatPhotons = cms.Sequence(
    photonMatch *
    patPhotons
    )
