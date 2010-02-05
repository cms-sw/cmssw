import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import *

makeLayer1METs = cms.Sequence(
    # reco pre-production
    patMETCorrections *
    # pat specifics
    # object production
    layer1METs
    )
