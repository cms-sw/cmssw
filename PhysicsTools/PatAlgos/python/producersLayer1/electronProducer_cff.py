import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.electronId_cff import *
from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import *

# add PAT specifics
from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *

makePatElectrons = cms.Sequence(
    # reco pre-production
    # patElectronId *
    # patElectronIsolation *
    # pat specifics
    electronMatch *
    # object production
    patElectrons
    )
