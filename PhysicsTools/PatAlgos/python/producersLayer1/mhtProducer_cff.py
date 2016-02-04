import FWCore.ParameterSet.Config as cms

# prepare reco information
# from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.mhtProducer_cfi import *

makePatMHTs = cms.Sequence(
    patMHTs
    )
