import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *

# add PAT specifics
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *

makeAllLayer1Jets = cms.Sequence(
    # reco pre-production
    patJetCharge *
    patJetCorrections *
    # pat specifics
    jetPartonMatch *
    jetGenJetMatch *
    jetFlavourId *
    # object production
    allLayer1Jets
    )
