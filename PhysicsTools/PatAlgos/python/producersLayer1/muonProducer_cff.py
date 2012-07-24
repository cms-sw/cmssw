import FWCore.ParameterSet.Config as cms

# prepare reco information
# NOT NEEDED AT THE MOMENT

# add PAT specifics
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

makePatMuons = cms.Sequence(
    # reco pre-production
    # pat specifics
    muonMatch *
    # object production
    patMuons
    )
