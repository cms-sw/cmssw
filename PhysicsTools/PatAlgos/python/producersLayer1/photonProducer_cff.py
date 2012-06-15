import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *

# add PAT specifics
from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *

makePatPhotons = cms.Sequence(
    # reco pre-production
    # patPhotonIsolation *
    # pat specifics
    photonMatch *
    # object production
    patPhotons
    )
