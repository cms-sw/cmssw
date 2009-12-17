import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import *

makePatMETs = cms.Sequence(
    # reco pre-production
    patMETCorrections *
    # pat specifics
    # object production
    patMETs
    )
